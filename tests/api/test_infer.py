from unittest.mock import Mock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
import builtins
import os


mock_model = Mock()
mock_model.predict.return_value = np.array([1])


original_open = builtins.open


def selective_mock_open(file, *args, **kwargs):
    if "model.pkl" in str(file):
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        return mock_file
    return original_open(file, *args, **kwargs)


with (
    patch("builtins.open", side_effect=selective_mock_open),
    patch("pickle.load", return_value=mock_model),
    patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False),
):
    from summit.api.infer import app


@pytest.fixture
def mock_infer_model():
    """Mock du modèle ML pour les tests."""
    model = Mock()
    model.predict.return_value = np.array([1])

    with patch("summit.api.infer.model", model):
        yield model


@pytest.fixture
def client():
    """Client de test sans authentification OAuth2 (OAUTH2_DOMAIN non défini)."""
    with patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False):
        yield TestClient(app)


def test_health_endpoint(client):
    """Test que le endpoint /health fonctionne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_infer_first_class_female(client, mock_infer_model):
    """Test prédiction pour une femme de 1ère classe."""
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_third_class_male(client, mock_infer_model):
    """Test prédiction pour un homme de 3ème classe."""
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([0])
    payload = {"pclass": 3, "sex": "male", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [0]
    mock_infer_model.predict.assert_called_once()


def test_infer_with_family(client, mock_infer_model):
    """Test prédiction avec des membres de la famille."""
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 2, "sex": "female", "sibSp": 1, "parch": 2}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_invalid_pclass(client):
    """Test validation avec une classe invalide."""
    payload = {"pclass": 5, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_invalid_sex(client):
    """Test validation avec un sexe invalide."""
    payload = {"pclass": 1, "sex": "unknown", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_missing_field(client):
    """Test avec un champ manquant."""
    payload = {"pclass": 1, "sex": "male", "sibSp": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_without_token(client):
    """Test que l'API refuse les requêtes sans token."""
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 401
