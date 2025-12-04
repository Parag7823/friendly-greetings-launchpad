"""
Airbyte Python Client - Async wrapper for Airbyte API
Replaces custom NangoClient implementation with Airbyte's built-in connectors.

This client handles:
- OAuth session creation for all providers
- Sync triggering and monitoring
- Connection management
- Retry logic with exponential backoff
- Prometheus metrics
- Structured logging
"""

import httpx
import time
from typing import Any, Dict, Optional, List
from prometheus_client import Counter, Histogram
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    AsyncRetrying,
)

from core_infrastructure.config_manager import get_airbyte_config

logger = structlog.get_logger(__name__)


class AirbytePythonClient:
    """
    Async client for Airbyte API.
    
    Replaces NangoClient by delegating OAuth and sync operations to Airbyte.
    Airbyte handles:
    - 300+ pre-built connectors
    - OAuth flows for all providers
    - Rate limiting per provider
    - Incremental sync with cursor management
    - Automatic deduplication
    - Retry logic with exponential backoff
    """

    # Prometheus metrics
    AIRBYTE_API_CALLS = Counter(
        'airbyte_api_calls_total',
        'Airbyte API calls',
        ['endpoint', 'method', 'status']
    )
    AIRBYTE_API_LATENCY = Histogram(
        'airbyte_api_latency_seconds',
        'Latency for Airbyte API calls',
        ['endpoint', 'method']
    )
    SYNC_TRIGGERED = Counter(
        'airbyte_syncs_triggered_total',
        'Airbyte syncs triggered',
        ['source', 'destination']
    )
    SYNC_COMPLETED = Counter(
        'airbyte_syncs_completed_total',
        'Airbyte syncs completed',
        ['source', 'status']
    )

    def __init__(self):
        """Initialize Airbyte client from centralized configuration."""
        config = get_airbyte_config()
        self.base_url = config.base_url  # e.g., http://localhost:8000/api/v1
        self.api_key = config.api_key
        self.workspace_id = config.workspace_id
        self.destination_id = config.destination_id  # Supabase destination
        
        # Timeouts for different operations
        self.default_timeout = config.default_timeout or 30.0
        self.oauth_timeout = config.oauth_timeout or 60.0
        self.sync_timeout = config.sync_timeout or 120.0
        self.status_timeout = config.status_timeout or 30.0

    def _headers(self) -> Dict[str, str]:
        """Build request headers with API key."""
        if not self.api_key:
            logger.error("❌ AIRBYTE_API_KEY is not set!")
        else:
            logger.info(f"✅ Airbyte API Key present (length: {len(self.api_key)})")
        
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        timeout: float,
        **kwargs
    ) -> httpx.Response:
        """
        Perform HTTP request with exponential backoff retry.
        
        Retries on:
        - Network errors
        - 429 (rate limit)
        - 5xx (server errors)
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
            retry=retry_if_exception_type((
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException
            )),
            reraise=True
        )
        
        async for attempt in retrying:
            with attempt:
                return await _make_request()

    async def create_oauth_session(
        self,
        provider: str,
        user_id: str,
        allowed_integrations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create OAuth session for user to authorize connection.
        
        Maps provider names to Airbyte source definitions:
        - 'gmail' -> 'source-gmail'
        - 'google-drive' -> 'source-google-drive'
        - 'dropbox' -> 'source-dropbox'
        - etc.
        
        Args:
            provider: Provider name (gmail, dropbox, google-drive, etc.)
            user_id: User ID for tracking
            allowed_integrations: List of allowed integration IDs (optional)
        
        Returns:
            Dict with OAuth session token and metadata
        """
        # Map provider names to Airbyte source definitions
        # CRITICAL FIX: Support both frontend naming conventions and Airbyte naming
        provider_to_source = {
            'gmail': 'source-gmail',
            'google-mail': 'source-gmail',  # Frontend sends 'google-mail', Airbyte expects 'gmail'
            'google-drive': 'source-google-drive',
            'dropbox': 'source-dropbox',
            'zoho-mail': 'source-zoho-mail',
            'quickbooks': 'source-quickbooks',
            'quickbooks-sandbox': 'source-quickbooks',  # Frontend sends 'quickbooks-sandbox', Airbyte expects 'quickbooks'
            'xero': 'source-xero',
            'stripe': 'source-stripe',
            'paypal': 'source-paypal-transaction',
            'razorpay': 'source-razorpay',
            'zoho-books': 'source-zoho-books',
        }
        
        source_id = provider_to_source.get(provider)
        if not source_id:
            raise ValueError(f"Unsupported provider: {provider}")
        
        url = f"{self.base_url}/sources/oauth_init"
        payload = {
            "source_definition_id": source_id,
            "workspace_id": self.workspace_id,
            "user_id": user_id,
        }
        
        logger.info(
            "airbyte_oauth_session_request",
            url=url,
            provider=provider,
            user_id=user_id
        )
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'POST',
                url,
                timeout=self.oauth_timeout,
                json=payload,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='oauth_init',
                method='POST',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='oauth_init',
                method='POST'
            ).observe(time.time() - t0)
            
            logger.info(
                "airbyte_oauth_session_response",
                status=resp.status_code,
                provider=provider
            )
            
            return resp.json()
        except Exception as e:
            logger.error(
                "airbyte_oauth_session_failed",
                error=str(e),
                provider=provider
            )
            raise

    async def trigger_sync(
        self,
        connection_id: str,
        provider: str,
        user_id: str,
        mode: str = "incremental"
    ) -> Dict[str, Any]:
        """
        Trigger a sync for a connection.
        
        Args:
            connection_id: Airbyte connection ID
            provider: Provider name (for logging)
            user_id: User ID (for logging)
            mode: 'incremental' or 'full_refresh'
        
        Returns:
            Dict with sync job ID and status
        """
        url = f"{self.base_url}/connections/{connection_id}/sync"
        payload = {
            "connectionId": connection_id,
            "syncMode": mode,
        }
        
        logger.info(
            "airbyte_sync_trigger_request",
            connection_id=connection_id,
            provider=provider,
            user_id=user_id,
            mode=mode
        )
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'POST',
                url,
                timeout=self.sync_timeout,
                json=payload,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='sync',
                method='POST',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='sync',
                method='POST'
            ).observe(time.time() - t0)
            
            result = resp.json()
            
            self.SYNC_TRIGGERED.labels(
                source=provider,
                destination='supabase'
            ).inc()
            
            logger.info(
                "airbyte_sync_triggered",
                connection_id=connection_id,
                job_id=result.get('job', {}).get('id'),
                provider=provider
            )
            
            return result
        except Exception as e:
            logger.error(
                "airbyte_sync_trigger_failed",
                error=str(e),
                connection_id=connection_id,
                provider=provider
            )
            raise

    async def get_sync_status(self, job_id: int) -> Dict[str, Any]:
        """
        Get status of a sync job.
        
        Args:
            job_id: Airbyte job ID
        
        Returns:
            Dict with job status, progress, errors, etc.
        """
        url = f"{self.base_url}/jobs/{job_id}"
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'GET',
                url,
                timeout=self.status_timeout,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='job_status',
                method='GET',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='job_status',
                method='GET'
            ).observe(time.time() - t0)
            
            result = resp.json()
            status = result.get('job', {}).get('status', 'unknown')
            
            logger.info(
                "airbyte_sync_status",
                job_id=job_id,
                status=status
            )
            
            return result
        except Exception as e:
            logger.error(
                "airbyte_sync_status_failed",
                error=str(e),
                job_id=job_id
            )
            raise

    async def create_connection(
        self,
        source_definition_id: str,
        destination_id: str,
        workspace_id: str,
        user_id: str,
        provider: str
    ) -> Dict[str, Any]:
        """
        Create an Airbyte connection (source → destination mapping).
        
        CRITICAL: Called after OAuth completes to create the connection
        that maps the authorized source to Supabase destination.
        
        Args:
            source_definition_id: Airbyte source definition ID (e.g., 'source-gmail')
            destination_id: Airbyte destination ID (Supabase)
            workspace_id: Airbyte workspace ID
            user_id: User ID (for logging)
            provider: Provider name (for logging)
        
        Returns:
            Dict with connection_id and status
        """
        url = f"{self.base_url}/connections"
        payload = {
            "sourceDefinitionId": source_definition_id,
            "destinationId": destination_id,
            "workspaceId": workspace_id,
            "name": f"{provider.capitalize()} Connection",
            "syncMode": "incremental",
            "status": "active"
        }
        
        logger.info(
            "airbyte_create_connection_request",
            url=url,
            provider=provider,
            user_id=user_id,
            source_definition_id=source_definition_id
        )
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'POST',
                url,
                timeout=self.default_timeout,
                json=payload,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='create_connection',
                method='POST',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='create_connection',
                method='POST'
            ).observe(time.time() - t0)
            
            result = resp.json()
            connection_id = result.get('connection', {}).get('connectionId')
            
            logger.info(
                "airbyte_connection_created",
                connection_id=connection_id,
                provider=provider,
                user_id=user_id
            )
            
            return result
        except Exception as e:
            logger.error(
                "airbyte_create_connection_failed",
                error=str(e),
                provider=provider,
                user_id=user_id
            )
            raise

    async def list_connections(self) -> Dict[str, Any]:
        """
        List all connections in workspace.
        
        Returns:
            Dict with list of connections
        """
        url = f"{self.base_url}/connections"
        params = {"workspaceId": self.workspace_id}
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'GET',
                url,
                timeout=self.default_timeout,
                params=params,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='connections',
                method='GET',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='connections',
                method='GET'
            ).observe(time.time() - t0)
            
            logger.info("airbyte_connections_listed")
            
            return resp.json()
        except Exception as e:
            logger.error("airbyte_list_connections_failed", error=str(e))
            raise

    async def delete_connection(self, connection_id: str) -> bool:
        """
        Delete a connection.
        
        Args:
            connection_id: Airbyte connection ID
        
        Returns:
            True if deletion succeeded
        """
        url = f"{self.base_url}/connections/{connection_id}"
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'DELETE',
                url,
                timeout=self.default_timeout,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='delete_connection',
                method='DELETE',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='delete_connection',
                method='DELETE'
            ).observe(time.time() - t0)
            
            if resp.status_code in (200, 202, 204):
                logger.info(
                    "airbyte_connection_deleted",
                    connection_id=connection_id
                )
                return True
            
            if resp.status_code == 404:
                logger.info(
                    "airbyte_connection_already_deleted",
                    connection_id=connection_id
                )
                return True
            
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(
                "airbyte_delete_connection_failed",
                error=str(e),
                connection_id=connection_id
            )
            return False

    async def get_source_schema(self, source_id: str) -> Dict[str, Any]:
        """
        Get schema for a source.
        
        Args:
            source_id: Airbyte source ID
        
        Returns:
            Dict with source schema
        """
        url = f"{self.base_url}/sources/{source_id}/schema_discovery"
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'GET',
                url,
                timeout=self.default_timeout,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='source_schema',
                method='GET',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='source_schema',
                method='GET'
            ).observe(time.time() - t0)
            
            logger.info("airbyte_source_schema_retrieved", source_id=source_id)
            
            return resp.json()
        except Exception as e:
            logger.error(
                "airbyte_get_source_schema_failed",
                error=str(e),
                source_id=source_id
            )
            raise

    async def get_destination_schema(self, destination_id: str) -> Dict[str, Any]:
        """
        Get schema for destination (Supabase).
        
        Args:
            destination_id: Airbyte destination ID
        
        Returns:
            Dict with destination schema
        """
        url = f"{self.base_url}/destinations/{destination_id}/schema_discovery"
        
        t0 = time.time()
        try:
            resp = await self._request_with_retry(
                'GET',
                url,
                timeout=self.default_timeout,
                headers=self._headers()
            )
            
            self.AIRBYTE_API_CALLS.labels(
                endpoint='destination_schema',
                method='GET',
                status=str(resp.status_code)
            ).inc()
            
            self.AIRBYTE_API_LATENCY.labels(
                endpoint='destination_schema',
                method='GET'
            ).observe(time.time() - t0)
            
            logger.info(
                "airbyte_destination_schema_retrieved",
                destination_id=destination_id
            )
            
            return resp.json()
        except Exception as e:
            logger.error(
                "airbyte_get_destination_schema_failed",
                error=str(e),
                destination_id=destination_id
            )
            raise
