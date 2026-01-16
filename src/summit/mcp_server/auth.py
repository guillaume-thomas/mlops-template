import os
import time
import logging
from typing import cast

import httpx


logger = logging.getLogger(__name__)


class OAuth2TokenManager:
    """Gestionnaire de tokens OAuth2 avec cache automatique."""

    def __init__(self) -> None:
        oauth2_domain = os.getenv("OAUTH2_DOMAIN")
        self.token_url = f"https://{oauth2_domain}/oauth/token" if oauth2_domain else None
        self.client_id = os.getenv("OAUTH2_CLIENT_ID")
        self.client_secret = os.getenv("OAUTH2_CLIENT_SECRET")
        self.scope = "api:read"

        self._access_token: str | None = None
        self._expires_at: float | None = None

        if not self.token_url or not self.client_id or not self.client_secret:
            logger.warning("OAuth2 credentials not configured, authentication will be skipped")

    def is_configured(self) -> bool:
        """Vérifie si OAuth2 est configuré."""
        return bool(self.token_url and self.client_id and self.client_secret)

    async def get_token(self) -> str | None:
        """Récupère un token valide (depuis le cache ou en le renouvelant)."""
        if not self.is_configured():
            return None

        if self._is_token_valid():
            return self._access_token

        return await self._refresh_token()

    def _is_token_valid(self) -> bool:
        """Vérifie si le token en cache est encore valide."""
        if not self._access_token or not self._expires_at:
            return False

        return time.time() < (self._expires_at - 60)

    async def _refresh_token(self) -> str:
        """Demande un nouveau token au serveur d'authentification."""
        if not self.token_url or not self.client_id or not self.client_secret:
            raise ValueError("OAuth2 credentials not configured")

        logger.info(f"Requesting new OAuth2 token from {self.token_url}")
        logger.info(f"Client ID: {self.client_id[:10]}...{self.client_id[-4:]}")
        logger.info(f"Scope: {self.scope}")
        logger.info("Audience: titanic-api")

        async with httpx.AsyncClient() as client:
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope,
                "audience": "titanic-api",
            }

            response = await client.post(
                self.token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            logger.info(f"OAuth2 response status: {response.status_code}")

            if response.status_code != 200:
                error_body = response.text
                logger.error("OAuth2 token request failed!")
                logger.error(f"Status code: {response.status_code}")
                logger.error(f"Response body: {error_body}")
                logger.error(f"Request URL: {self.token_url}")
                response.raise_for_status()

            data = response.json()

            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._expires_at = time.time() + expires_in

            logger.info(f"OAuth2 token refreshed successfully, expires in {expires_in}s")
            return cast(str, self._access_token)


token_manager = OAuth2TokenManager()
