from unittest.mock import patch, Mock

from summit.ci.search_mlflow import get_last_model_uri


def test_get_last_model_uri_returns_uri():
    with patch("summit.ci.search_mlflow.mlflow") as mock_mlflow:
        mock_experiment = {"experiment_id": "exp-123"}
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run_info = Mock()
        mock_run_info.info.run_id = "run-456"
        mock_mlflow.search_runs.return_value = [mock_run_info]

        mock_run = Mock()
        mock_model_output = Mock()
        mock_model_output.model_id = "model-789"
        mock_run.outputs.model_outputs = [mock_model_output]
        mock_mlflow.get_run.return_value = mock_run

        result = get_last_model_uri("test-exp")

        assert result == "models:/model-789"
