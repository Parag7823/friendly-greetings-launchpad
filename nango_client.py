import os
import base64
from typing import Any, Dict, Optional

import httpx


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
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=self._headers(provider_config_key, connection_id))
            resp.raise_for_status()
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, params=params, headers=self._headers(provider_config_key, connection_id))
            resp.raise_for_status()
            return resp.json()

    async def get_gmail_message(self, provider_config_key: str, connection_id: str, message_id: str) -> Dict[str, Any]:
        """Get full Gmail message to locate attachment parts."""
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages/{message_id}"
        params = {"format": "full"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, params=params, headers=self._headers(provider_config_key, connection_id))
            resp.raise_for_status()
            return resp.json()

    async def get_gmail_attachment(self, provider_config_key: str, connection_id: str,
                                   message_id: str, attachment_id: str) -> bytes:
        """Fetch a Gmail attachment as bytes (base64 decode)."""
        url = f"{self.base_url}/proxy/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url, headers=self._headers(provider_config_key, connection_id))
            resp.raise_for_status()
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
