from unittest.mock import patch, Mock, AsyncMock
from summit.mcp_server.server import mcp, predict_survival, health_check
import pytest
from starlette.requests import Request


def test_mcp_server_configuration():
    """Test que le serveur MCP est correctement configuré."""
    assert mcp is not None
    assert hasattr(mcp, "name")
    assert mcp.name == "titanic-mcp-server"


@pytest.mark.asyncio
async def test_predict_survival_with_successful_api_call():
    """Test predict_survival avec une API qui retourne survived."""
    mock_response = AsyncMock()
    mock_response.json = lambda: [1]
    mock_response.raise_for_status = lambda: None

    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("summit.mcp_server.server.token_manager.get_token", return_value=None),
    ):
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        result = await predict_survival.fn(pclass=1, sex="female", sibsp=0, parch=0)

        assert "SURVIVED" in result
        assert "Good news" in result


@pytest.mark.asyncio
async def test_predict_survival_with_death_prediction():
    """Test predict_survival avec une API qui retourne not survived."""
    mock_response = AsyncMock()
    mock_response.json = lambda: [0]
    mock_response.raise_for_status = lambda: None

    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("summit.mcp_server.server.token_manager.get_token", return_value=None),
    ):
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        result = await predict_survival.fn(pclass=3, sex="male", sibsp=0, parch=0)

        assert "NOT have survived" in result
        assert "Unfortunately" in result


@pytest.mark.asyncio
async def test_predict_survival_with_oauth2_token():
    """Test que predict_survival utilise le token OAuth2 quand disponible."""
    mock_response = AsyncMock()
    mock_response.json = lambda: [1]
    mock_response.raise_for_status = lambda: None

    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("summit.mcp_server.server.token_manager.get_token", return_value="test-token-123") as mock_get_token,
    ):
        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        result = await predict_survival.fn(pclass=1, sex="female", sibsp=0, parch=0)

        assert "SURVIVED" in result
        mock_get_token.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token-123"


@pytest.mark.asyncio
async def test_predict_survival_handles_api_errors():
    """Test que predict_survival gère gracieusement les erreurs API."""
    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("summit.mcp_server.server.token_manager.get_token", return_value=None),
    ):
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=Exception("Connection timeout"))

        result = await predict_survival.fn(pclass=1, sex="female", sibsp=0, parch=0)

        assert "error" in result.lower()
        assert "Connection timeout" in result


@pytest.mark.asyncio
async def test_health_check_returns_healthy():
    """Test que le health check retourne le bon statut."""

    mock_request = Mock(spec=Request)
    response = await health_check(mock_request)

    assert response.status_code == 200
    assert b"healthy" in response.body
