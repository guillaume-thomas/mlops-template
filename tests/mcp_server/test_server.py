from unittest.mock import patch, Mock
from summit.mcp_server.server import mcp, predict_survival, health_check
import requests
import pytest
from starlette.requests import Request


def test_mcp_server_configuration():
    """Test que le serveur MCP est correctement configuré."""
    assert mcp is not None
    assert hasattr(mcp, "name")
    assert mcp.name == "titanic-mcp-server"


def test_predict_survival_with_successful_api_call():
    """Test predict_survival avec une API qui retourne survived."""
    with patch.object(requests, "post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = [1]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = predict_survival.fn(pclass=1, sex="female", sibsp=0, parch=0)

        assert "SURVIVED" in result
        assert "Good news" in result
        mock_post.assert_called_once()


def test_predict_survival_with_death_prediction():
    """Test predict_survival avec une API qui retourne not survived."""
    with patch.object(requests, "post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = [0]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = predict_survival.fn(pclass=3, sex="male", sibsp=0, parch=0)

        assert "NOT have survived" in result
        assert "Unfortunately" in result


def test_predict_survival_handles_api_errors():
    """Test que predict_survival gère gracieusement les erreurs API."""
    with patch.object(requests, "post") as mock_post:
        mock_post.side_effect = Exception("Connection timeout")

        result = predict_survival.fn(pclass=1, sex="female", sibsp=0, parch=0)

        assert "error" in result.lower()
        assert "Connection timeout" in result


@pytest.mark.asyncio
async def test_health_check_returns_healthy():
    """Test que le health check retourne le bon statut."""

    mock_request = Mock(spec=Request)
    response = await health_check(mock_request)

    assert response.status_code == 200
    assert b"healthy" in response.body
