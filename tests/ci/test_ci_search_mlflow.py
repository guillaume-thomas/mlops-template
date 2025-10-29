import pytest
from unittest.mock import patch, MagicMock

from summit.ci.search_mlflow import get_last_model_uri


class TestSearchMlflow:

    @patch('summit.ci.search_mlflow.mlflow')
    def test_get_last_model_uri_success(self, mock_mlflow):
        experiment_name = "test_experiment"

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

        result = get_last_model_uri(experiment_name)

        assert result == "models:/model_789"
        mock_mlflow.get_experiment_by_name.assert_called_once_with(experiment_name)
        mock_mlflow.search_runs.assert_called_once_with(
            ["123"],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1,
            order_by=["attributes.end_time DESC"],
            output_format="list"
        )
        mock_mlflow.get_run.assert_called_once_with("run_456")

    @patch('summit.ci.search_mlflow.mlflow')
    @patch('summit.ci.search_mlflow.logging')
    def test_get_last_model_uri_logs_experiment_name(self, mock_logging, mock_mlflow):
        experiment_name = "test_experiment"

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

        get_last_model_uri(experiment_name)

        mock_logging.warning.assert_any_call(f"experiment_name: {experiment_name}")

    @patch('summit.ci.search_mlflow.mlflow')
    @patch('summit.ci.search_mlflow.logging')
    def test_get_last_model_uri_logs_model_info(self, mock_logging, mock_mlflow):
        experiment_name = "test_experiment"

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

        get_last_model_uri(experiment_name)

        mock_logging.warning.assert_any_call("Found model id: model_789")
        mock_logging.warning.assert_any_call("Returning: models:/model_789")

    @patch('summit.ci.search_mlflow.mlflow')
    def test_get_last_model_uri_with_different_experiment_id(self, mock_mlflow):
        experiment_name = "another_experiment"

        mock_experiment = {"experiment_id": "999"}
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run_info = MagicMock()
        mock_run_info.run_id = "run_999"
        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_mlflow.search_runs.return_value = [mock_run]

        mock_complete_run = MagicMock()
        mock_model_output = MagicMock()
        mock_model_output.model_id = "model_999"
        mock_complete_run.outputs.model_outputs = [mock_model_output]
        mock_mlflow.get_run.return_value = mock_complete_run

        result = get_last_model_uri(experiment_name)

        assert result == "models:/model_999"
        mock_mlflow.search_runs.assert_called_once_with(
            ["999"],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1,
            order_by=["attributes.end_time DESC"],
            output_format="list"
        )

    @patch('summit.ci.search_mlflow.mlflow')
    def test_get_last_model_uri_no_experiments_found(self, mock_mlflow):
        experiment_name = "nonexistent_experiment"

        mock_mlflow.get_experiment_by_name.return_value = None

        with pytest.raises(TypeError):
            get_last_model_uri(experiment_name)

    @patch('summit.ci.search_mlflow.mlflow')
    def test_get_last_model_uri_no_runs_found(self, mock_mlflow):
        experiment_name = "test_experiment"

        mock_experiment = {"experiment_id": "123"}
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = []

        with pytest.raises(IndexError):
            get_last_model_uri(experiment_name)

    def test_function_exists(self):
        assert callable(get_last_model_uri)
