from unittest.mock import patch
from fastapi.testclient import TestClient
import numpy as np

from summit.api.infer import app, Something, health, infer


class TestApiInfer:

    def setup_method(self):
        self.client = TestClient(app)

    def test_model_loading_mock_behavior(self):
        # Ce test vérifie que le modèle peut être mocké correctement
        with patch('summit.api.infer.model') as mock_model:
            mock_model.predict.return_value = [1.0]

            # Test que le mock fonctionne
            result = mock_model.predict([[1, 2, 3]])
            assert result == [1.0]


    def test_something_dataclass_structure(self):
        something = Something(A=1, B=2, C=3)
        assert something.A == 1
        assert something.B == 2
        assert something.C == 3

    def test_something_dataclass_types(self):
        something = Something(A=10, B=20, C=30)
        assert isinstance(something.A, int)
        assert isinstance(something.B, int)
        assert isinstance(something.C, int)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    def test_health_function_direct_call(self):
        result = health()
        assert result == {"status": "OK"}

    @patch('summit.api.infer.model')
    def test_infer_endpoint_success(self, mock_model):
        mock_model.predict.return_value = np.array([1.5])

        response = self.client.post("/infer", json={"A": 1, "B": 2, "C": 3})

        assert response.status_code == 200
        assert response.json() == [1.5]
        mock_model.predict.assert_called_once_with([[1, 2, 3]])

    @patch('summit.api.infer.model')
    def test_infer_function_direct_call(self, mock_model):
        mock_model.predict.return_value = np.array([2.5, 3.5])

        something = Something(A=5, B=10, C=15)
        result = infer(something)

        assert result == [2.5, 3.5]
        mock_model.predict.assert_called_once_with([[5, 10, 15]])

    @patch('summit.api.infer.model')
    def test_infer_with_different_values(self, mock_model):
        mock_model.predict.return_value = np.array([0.1, 0.9])

        something = Something(A=100, B=200, C=300)
        result = infer(something)

        assert result == [0.1, 0.9]
        mock_model.predict.assert_called_once_with([[100, 200, 300]])

    def test_infer_endpoint_invalid_data(self):
        response = self.client.post("/infer", json={"A": "invalid", "B": 2, "C": 3})
        assert response.status_code == 422

    def test_infer_endpoint_missing_fields(self):
        response = self.client.post("/infer", json={"A": 1, "B": 2})
        assert response.status_code == 422

    @patch('summit.api.infer.model')
    def test_infer_returns_list(self, mock_model):
        mock_model.predict.return_value = np.array([1, 2, 3])

        something = Something(A=1, B=2, C=3)
        result = infer(something)

        assert isinstance(result, list)
        assert result == [1, 2, 3]
