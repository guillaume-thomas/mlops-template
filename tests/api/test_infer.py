from fastapi.testclient import TestClient
import pytest

from summit.api.infer import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Test que le endpoint /health fonctionne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_infer_first_class_female(client):
    """Test prédiction pour une femme de 1ère classe."""
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in [0, 1]


def test_infer_third_class_male(client):
    """Test prédiction pour un homme de 3ème classe."""
    payload = {"pclass": 3, "sex": "male", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in [0, 1]


def test_infer_with_family(client):
    """Test prédiction avec des membres de la famille."""
    payload = {"pclass": 2, "sex": "female", "sibSp": 1, "parch": 2}
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1


def test_infer_invalid_pclass(client):
    """Test validation avec une classe invalide."""
    payload = {"pclass": 5, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 422


def test_infer_invalid_sex(client):
    """Test validation avec un sexe invalide."""
    payload = {"pclass": 1, "sex": "unknown", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 422


def test_infer_missing_field(client):
    """Test avec un champ manquant."""
    payload = {"pclass": 1, "sex": "male", "sibSp": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 422
