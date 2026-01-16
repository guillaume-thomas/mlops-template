import pytest
from unittest.mock import patch, AsyncMock
import time

from summit.mcp_server.auth import OAuth2TokenManager


@pytest.fixture
def mock_env_oauth2(monkeypatch):
    """Configure les variables d'environnement OAuth2 pour les tests."""
    monkeypatch.setenv("OAUTH2_DOMAIN", "auth.example.com")
    monkeypatch.setenv("OAUTH2_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("OAUTH2_CLIENT_SECRET", "test-client-secret")


def test_token_manager_is_configured(mock_env_oauth2):
    """Test que le token manager détecte correctement la configuration."""
    manager = OAuth2TokenManager()
    assert manager.is_configured() is True


def test_token_manager_not_configured_without_env():
    """Test que le token manager détecte l'absence de configuration."""
    manager = OAuth2TokenManager()
    assert manager.is_configured() is False


@pytest.mark.asyncio
async def test_get_token_returns_none_when_not_configured():
    """Test que get_token retourne None quand OAuth2 n'est pas configuré."""
    manager = OAuth2TokenManager()
    token = await manager.get_token()
    assert token is None


@pytest.mark.asyncio
async def test_get_token_fetches_new_token(mock_env_oauth2):
    """Test que get_token récupère un nouveau token."""
    manager = OAuth2TokenManager()

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json = lambda: {"access_token": "test-token-123", "expires_in": 3600}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        token = await manager.get_token()

        assert token == "test-token-123"
        assert manager._access_token == "test-token-123"
        assert manager._expires_at is not None


@pytest.mark.asyncio
async def test_get_token_uses_cache(mock_env_oauth2):
    """Test que get_token utilise le cache quand le token est valide."""
    manager = OAuth2TokenManager()
    manager._access_token = "cached-token"
    manager._expires_at = time.time() + 3600

    token = await manager.get_token()

    assert token == "cached-token"


@pytest.mark.asyncio
async def test_token_refresh_when_expired(mock_env_oauth2):
    """Test que le token est renouvelé quand il est expiré."""
    manager = OAuth2TokenManager()
    manager._access_token = "old-token"
    manager._expires_at = time.time() - 100

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json = lambda: {"access_token": "new-token-456", "expires_in": 3600}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        token = await manager.get_token()

        assert token == "new-token-456"
        assert manager._access_token == "new-token-456"
