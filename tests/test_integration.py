from unittest.mock import patch, MagicMock
import numpy as np



class TestIntegration:

    def test_all_modules_importable(self):
        """Test que tous les modules du package summit peuvent être importés"""
        import summit.main
        import summit.api.main
        import summit.api.infer
        import summit.ci.search_mlflow
        import summit.training.main
        import summit.training.steps.load_data
        import summit.training.steps.split_train_test
        import summit.training.steps.train
        import summit.training.steps.validate
        import summit.monitoring

        assert True

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    def test_training_workflow_integration(self, mock_mlflow, mock_load_data,
                                          mock_split_train_test, mock_train, mock_validate):
        """Test d'intégration du workflow de training complet"""
        from summit.training.main import workflow

        mock_context_manager = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = mock_context_manager
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_load_data.return_value = "data_path"
        mock_split_train_test.return_value = ("xtrain", "xtest", "ytrain", "ytest")
        mock_train.return_value = "model_path"

        workflow("/path/to/data.csv")

        mock_load_data.assert_called_once()
        mock_split_train_test.assert_called_once()
        mock_train.assert_called_once()
        mock_validate.assert_called_once()

    @patch('summit.api.infer.model')
    def test_api_integration(self, mock_model):
        """Test d'intégration de l'API"""
        from fastapi.testclient import TestClient
        from summit.api.infer import app, Something

        import numpy as np
        mock_model.predict.return_value = np.array([1.0, 2.0])

        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

        response = client.post("/infer", json={"A": 1, "B": 2, "C": 3})
        assert response.status_code == 200
        assert response.json() == [1.0, 2.0]

    @patch('summit.ci.search_mlflow.mlflow')
    def test_ci_integration(self, mock_mlflow):
        """Test d'intégration du système CI/CD avec MLflow"""
        from summit.ci.search_mlflow import get_last_model_uri

        mock_experiment = {"experiment_id": "123"}
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run_info = MagicMock()
        mock_run_info.run_id = "run_456"
        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_mlflow.search_runs.return_value = [mock_run]

        mock_complete_run = MagicMock()
        mock_model_output = MagicMock()
        mock_model_output.model_id = "model_789"
        mock_complete_run.outputs.model_outputs = [mock_model_output]
        mock_mlflow.get_run.return_value = mock_complete_run

        result = get_last_model_uri("test_experiment")
        assert result == "models:/model_789"

    def test_package_structure(self):
        """Test que la structure du package est correcte"""
        import summit
        import summit.api
        import summit.training
        import summit.training.steps
        import summit.ci
        import summit.monitoring

        assert hasattr(summit, 'main')
        assert hasattr(summit.api, 'main')
        assert hasattr(summit.api, 'infer')
        assert hasattr(summit.training, 'main')
        assert hasattr(summit.training.steps, 'load_data')
        assert hasattr(summit.training.steps, 'split_train_test')
        assert hasattr(summit.training.steps, 'train')
        assert hasattr(summit.training.steps, 'validate')
        assert hasattr(summit.ci, 'search_mlflow')

    def test_main_functions_callable(self):
        """Test que toutes les fonctions principales sont appelables"""
        from summit.main import main as summit_main
        from summit.api.main import main as api_main
        from summit.training.main import workflow
        from summit.ci.search_mlflow import get_last_model_uri
        from summit.training.steps.load_data import load_data
        from summit.training.steps.split_train_test import split_train_test
        from summit.training.steps.train import train
        from summit.training.steps.validate import validate

        assert callable(summit_main)
        assert callable(api_main)
        assert callable(workflow)
        assert callable(get_last_model_uri)
        assert callable(load_data)
        assert callable(split_train_test)
        assert callable(train)
        assert callable(validate)
