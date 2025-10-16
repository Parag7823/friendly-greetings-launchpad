import os
import base64
from typing import Any, Dict, Optional

import httpx
import time
import asyncio
from prometheus_client import Counter, Histogram


class NangoClient:
    """Thin async client for calling Nango's hosted API in dev/prod.

    Uses the Proxy to hit underlying provider APIs (Gmail here) and the Connect Session API
    to generate session tokens for the hosted auth UI.
    """

    def __init__(self, base_url: Optional[str] = None, secret_key: Optional[str] = None):
        self.base_url = base_url or os.environ.get("NANGO_BASE_URL", "https://api.nango.dev")
        self.secret_key = secret_key or os.environ.get("NANGO_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("NANGO_SECRET_KEY env var not set")

    def _headers(self, provider_config_key: Optional[str] = None, connection_id: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.secret_key}",
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
        """
        url = f"{self.base_url}/connect/sessions"
        payload = {
            "end_user": end_user,
            "allowed_integrations": allowed_integrations,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
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
            'GET', url, timeout=30.0,
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
            'GET', url, timeout=60.0,
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
            'GET', url, timeout=60.0,
            params=params, headers=self._headers(provider_config_key, connection_id)
        )
        return resp.json()

    async def get_gmail_attachment(self, provider_config_key: str, connection_id: str,
                                   message_id: str, attachment_id: str) -> bytes:
        """Fetch a Gmail attachment as bytes (base64 decode)."""
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}"
        _PROVIDER = 'gmail'
        _METHOD = 'GET'
        t0 = None
        try:
            t0 = time.time()
        except Exception:
            pass
        resp = await self._request_with_retry(
            'GET', url, timeout=120.0,
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
            return base64.b64decode(b64)
        except Exception:
            # Try urlsafe decode as fallback
            return base64.urlsafe_b64decode(b64)

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
            'GET', url, timeout=60.0,
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

    async def _request_with_retry(self, method: str, url: str, timeout: float, max_attempts: int = 3, backoff_base: float = 1.0, **kwargs) -> httpx.Response:
        """Perform an HTTP request with simple exponential backoff on transient errors.

        Retries on network errors and 429/5xx HTTP responses.
        """
        attempt = 0
        last_exc: Exception | None = None
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(method, url, **kwargs)
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        code = e.response.status_code if e.response is not None else 0
                        if code == 429 or 500 <= code < 600:
                            raise
                        # Non-retryable
                        raise
                    return resp
            except httpx.HTTPStatusError as e:
                last_exc = e
            except httpx.RequestError as e:
                # Includes timeouts, connection errors, read errors (e.g., wsarecv on Windows)
                last_exc = e
            except Exception as e:
                last_exc = e
            attempt += 1
            # Exponential backoff with jitter
            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)) + (0.1 * attempt))
        # Exhausted retries
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed with unknown error")

    async def proxy_get(self, provider: str, path: str, params: dict | None = None,
                        connection_id: str | None = None, provider_config_key: str | None = None,
                        headers: dict | None = None) -> dict:
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

    async def proxy_get_bytes(self, provider: str, path: str, params: dict | None = None,
                              connection_id: str | None = None, provider_config_key: str | None = None) -> bytes:
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

    async def proxy_post(self, provider: str, path: str, json_body: dict | None = None,
                         connection_id: str | None = None, provider_config_key: str | None = None,
                         headers: dict | None = None) -> dict:
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
