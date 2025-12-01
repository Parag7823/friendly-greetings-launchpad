import base64
from typing import Any, Dict, Optional

import httpx
import time
from prometheus_client import Counter, Histogram
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    AsyncRetrying,
)

from core_infrastructure.config_manager import get_nango_config, get_app_config

logger = structlog.get_logger(__name__)


class NangoClient:
    """Thin async client for calling Nango's hosted API in dev/prod.

    Uses the Proxy to hit underlying provider APIs (Gmail here) and the Connect Session API
    to generate session tokens for the hosted auth UI.
    
    Retry logic is handled by tenacity library with exponential backoff.
    """

    def __init__(self):
        """Initialize Nango client from centralized configuration."""
        config = get_nango_config()
        self.base_url = config.base_url
        self.secret_key = config.secret_key
        
        # Store configurable timeouts from config
        self.default_timeout = config.default_timeout
        self.connect_session_timeout = config.connect_session_timeout
        self.gmail_profile_timeout = config.gmail_profile_timeout
        self.gmail_list_timeout = config.gmail_list_timeout
        self.gmail_attachment_timeout = config.gmail_attachment_timeout
        self.gmail_history_timeout = config.gmail_history_timeout

    def _headers(self, provider_config_key: Optional[str] = None, connection_id: Optional[str] = None) -> Dict[str, str]:
        if not self.secret_key:
            logger.error("❌ NANGO_SECRET_KEY is not set!")
        else:
            logger.info(f"✅ Nango API Key present (length: {len(self.secret_key)})")
        
        headers = {
            'Authorization': f'Bearer {self.secret_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if provider_config_key:
            headers["Provider-Config-Key"] = provider_config_key
        if connection_id:
            headers["Connection-Id"] = connection_id
            
        return headers

    async def create_connect_session(self, end_user: Dict[str, Any], allowed_integrations: list[str]) -> Dict[str, Any]:
        """Create a Nango Connect session token to authorize a connection via hosted UI.

        API: POST /connect/sessions
        Docs: https://docs.nango.dev/reference/api/connect/sessions/create
        
        Args:
            end_user: Dict with 'id' key containing user identifier
            allowed_integrations: List of integration IDs (strings) e.g. ["google-drive", "gmail"]
                                 NOT objects like [{"provider_config_key": "google-drive"}]
        
        Returns:
            Dict containing session token and metadata
        """
        url = f"{self.base_url}/connect/sessions"
        payload = {
            "end_user": end_user,
            "allowed_integrations": allowed_integrations,
        }
        
        # Use structlog logger (already imported at module level)
        logger.info("nango_connect_session_request", url=url, integrations=allowed_integrations)
        
        async with httpx.AsyncClient(timeout=self.connect_session_timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            
            logger.info("nango_connect_session_response", status=resp.status_code)
            
            resp.raise_for_status()
            return resp.json()

    # ------------------------- Gmail via Proxy -------------------------
    async def get_gmail_profile(self, provider_config_key: str, connection_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/proxy/gmail/v1/users/me/profile"
        _PROVIDER = 'gmail'
        _METHOD = 'GET'
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=self.gmail_profile_timeout,
            headers=self._headers(provider_config_key, connection_id)
        )
        try:
            self.NANGO_API_CALLS.labels(provider=_PROVIDER, method=_METHOD, status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=_PROVIDER, method=_METHOD).observe(time.time() - t0)
        except Exception:
            pass
        return resp.json()

    async def list_gmail_messages(self, provider_config_key: str, connection_id: str, q: str,
                                  page_token: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
        """List Gmail messages matching query. Uses Gmail REST via Nango Proxy.

        q examples: 'has:attachment newer_than:365d'
        """
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages"
        params = {"q": q, "maxResults": max_results}
        if page_token:
            params["pageToken"] = page_token
        _PROVIDER = 'gmail'
        _METHOD = 'GET'
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=self.gmail_list_timeout,
            params=params, headers=self._headers(provider_config_key, connection_id)
        )
        try:
            self.NANGO_API_CALLS.labels(provider=_PROVIDER, method=_METHOD, status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=_PROVIDER, method=_METHOD).observe(time.time() - t0)
        except Exception:
            pass
        return resp.json()

    async def get_gmail_message(self, provider_config_key: str, connection_id: str, message_id: str) -> Dict[str, Any]:
        """Get full Gmail message to locate attachment parts."""
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages/{message_id}"
        params = {"format": "full"}
        resp = await self._request_with_retry(
            'GET', url, timeout=self.gmail_list_timeout,
            params=params, headers=self._headers(provider_config_key, connection_id)
        )
        return resp.json()

    async def get_gmail_attachment(self, provider_config_key: str, connection_id: str,
                                   message_id: str, attachment_id: str) -> Optional[bytes]:
        """
        Fetch a Gmail attachment as bytes (base64 decode).
        
        CRITICAL FIX #7: Added size check to prevent OOM crashes.
        Returns None if attachment exceeds max_attachment_size_mb.
        """
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}"
        _PROVIDER = 'gmail'
        _METHOD = 'GET'
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=self.gmail_attachment_timeout,
            headers=self._headers(provider_config_key, connection_id)
        )
        try:
            self.NANGO_API_CALLS.labels(provider=_PROVIDER, method=_METHOD, status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=_PROVIDER, method=_METHOD).observe(time.time() - t0)
        except Exception:
            pass
        data = resp.json()
        # Gmail returns base64url data under 'data'
        b64 = data.get("data")
        if not b64:
            return b""
        # Gmail API uses base64url; handle both
        b64 = b64.replace("-", "+").replace("_", "/")
        try:
            decoded = base64.b64decode(b64)
        except Exception:
            # Try urlsafe decode as fallback
            decoded = base64.urlsafe_b64decode(b64)
        
        # CRITICAL FIX #7: Check attachment size before returning
        app_config = get_app_config()
        max_size_bytes = app_config.max_attachment_size_mb * 1024 * 1024
        
        if len(decoded) > max_size_bytes:
            logger.warning(
                f"Attachment too large: {len(decoded)} bytes > {max_size_bytes} bytes limit. "
                f"Skipping attachment {attachment_id} from message {message_id}"
            )
            return None
        
        return decoded

    async def list_gmail_history(self, provider_config_key: str, connection_id: str, 
                                 start_history_id: str, max_results: int = 100,
                                 page_token: Optional[str] = None) -> Dict[str, Any]:
        """List Gmail history changes since a given historyId.
        
        This is the true incremental sync API that returns only changes (new messages, 
        deleted messages, label changes) since the last sync.
        
        API: GET /gmail/v1/users/me/history
        Docs: https://developers.google.com/gmail/api/reference/rest/v1/users.history/list
        
        Args:
            provider_config_key: Nango provider config key
            connection_id: Nango connection ID
            start_history_id: History ID to start from (from previous sync)
            max_results: Maximum number of history records to return (default 100)
            page_token: Token for pagination (optional)
            
        Returns:
            Dict containing:
                - history: List of history records
                - historyId: Current history ID (save for next sync)
                - nextPageToken: Token for next page (if more results exist)
        """
        url = f"{self.base_url}/proxy/gmail/v1/users/me/history"
        params = {
            "startHistoryId": start_history_id,
            "maxResults": max_results,
            "historyTypes": ["messageAdded"]  # Only track new messages with attachments
        }
        if page_token:
            params["pageToken"] = page_token
            
        _PROVIDER = 'gmail'
        _METHOD = 'GET'
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
            
        resp = await self._request_with_retry(
            'GET', url, timeout=self.gmail_history_timeout,
            params=params, headers=self._headers(provider_config_key, connection_id)
        )
        
        try:
            self.NANGO_API_CALLS.labels(provider=_PROVIDER, method=_METHOD, status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=_PROVIDER, method=_METHOD).observe(time.time() - t0)
        except Exception:
            pass
            
        return resp.json()

    # ------------------------- Generic Proxy Helpers -------------------------
    # Prometheus metrics
    NANGO_API_CALLS = Counter('nango_api_calls_total', 'Nango proxy API calls', ['provider', 'method', 'status'])
    NANGO_API_LATENCY = Histogram('nango_api_latency_seconds', 'Latency for Nango proxy API calls', ['provider', 'method'])

    async def _request_with_retry(self, method: str, url: str, timeout: float, **kwargs) -> httpx.Response:
        """Perform an HTTP request with exponential backoff retry using tenacity.

        Retries on network errors and 429/5xx HTTP responses.
        Tenacity handles all retry logic, backoff, and jitter automatically.
        """
        async def _make_request():
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, **kwargs)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code if e.response is not None else 0
                    # Retry on rate limit (429) and server errors (5xx)
                    if code == 429 or 500 <= code < 600:
                        raise
                    # Non-retryable errors (4xx except 429)
                    raise
                return resp

        # Use tenacity for retry logic with exponential backoff
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)),
            reraise=True
        )
        
        async for attempt in retrying:
            with attempt:
                return await _make_request()

    async def proxy_get(
        self,
        provider: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        connection_id: Optional[str] = None,
        provider_config_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Perform a GET via Nango Proxy and parse JSON response.

        Example: await client.proxy_get('google-drive', 'drive/v3/files', params={...}, ...)
        """
        url = f"{self.base_url}/proxy/{provider}/{path.lstrip('/')}"
        merged_headers = self._headers(provider_config_key, connection_id)
        if headers:
            merged_headers.update(headers)
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=60.0,
            params=params or {}, headers=merged_headers
        )
        try:
            self.NANGO_API_CALLS.labels(provider=provider, method='GET', status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=provider, method='GET').observe(time.time() - t0)
        except Exception:
            pass
        return resp.json()

    async def proxy_get_bytes(
        self,
        provider: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        connection_id: Optional[str] = None,
        provider_config_key: Optional[str] = None,
    ) -> bytes:
        """Perform a GET via Nango Proxy and return raw bytes (for media endpoints)."""
        url = f"{self.base_url}/proxy/{provider}/{path.lstrip('/')}"
        merged_headers = self._headers(provider_config_key, connection_id)
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=120.0,
            params=params or {}, headers=merged_headers
        )
        try:
            self.NANGO_API_CALLS.labels(provider=provider, method='GET', status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=provider, method='GET').observe(time.time() - t0)
        except Exception:
            pass
        return resp.content

    async def proxy_post(
        self,
        provider: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        connection_id: Optional[str] = None,
        provider_config_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Perform a POST via Nango Proxy.

        If the response body is not JSON, return a dict with {'_raw': bytes}.
        """
        url = f"{self.base_url}/proxy/{provider}/{path.lstrip('/')}"
        merged_headers = self._headers(provider_config_key, connection_id)
        if headers:
            merged_headers.update(headers)
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'POST', url, timeout=120.0,
            json=json_body, headers=merged_headers
        )
        try:
            self.NANGO_API_CALLS.labels(provider=provider, method='POST', status=str(resp.status_code)).inc()
            if t0 is not None:
                self.NANGO_API_LATENCY.labels(provider=provider, method='POST').observe(time.time() - t0)
        except Exception:
            pass
        # Try JSON first
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.content}

    async def delete_connection(self, connection_id: str, provider_config_key: Optional[str] = None) -> bool:
        """Delete a Nango connection if it exists.

        Args:
            connection_id: The Nango connection identifier (usually {user_id}_{integration_id}).
            provider_config_key: Optional provider config key / integration identifier.

        Returns:
            True if deletion succeeded or the connection was already gone. False otherwise.
        """
        url = f"{self.base_url}/connection/{connection_id}"
        params: Dict[str, Any] = {}
        if provider_config_key:
            params['provider_config_key'] = provider_config_key

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    url,
                    params=params,
                    headers=self._headers(provider_config_key, connection_id)
                )

            if resp.status_code in (200, 202, 204):
                return True

            if resp.status_code == 404:
                # Connection already removed – treat as success
                return True

            resp.raise_for_status()
            return True
        except Exception:
            # Let caller decide how to handle failures; keep surface area small here
            return False
