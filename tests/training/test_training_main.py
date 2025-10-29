import pytest
from unittest.mock import patch, MagicMock

from summit.training.main import workflow


class TestTrainingMain:

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    @patch('summit.training.main.logging')
    def test_workflow_complete_pipeline(self, mock_logging, mock_mlflow, mock_load_data,
                                       mock_split_train_test, mock_train, mock_validate):
        input_data_path = "/path/to/input/data.csv"

        mock_context_manager = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = mock_context_manager
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_load_data.return_value = "processed_data_path"
        mock_split_train_test.return_value = ("xtrain_path", "xtest_path", "ytrain_path", "ytest_path")
        mock_train.return_value = "model_path"

        workflow(input_data_path)

        mock_logging.warning.assert_called_with(f"workflow input path : {input_data_path}")
        mock_mlflow.start_run.assert_called_once()
        mock_load_data.assert_called_once_with(input_data_path)
        mock_split_train_test.assert_called_once_with("processed_data_path")
        mock_train.assert_called_once_with("xtrain_path", "ytrain_path")
        mock_validate.assert_called_once_with("model_path", "xtest_path", "ytest_path")

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    def test_workflow_with_different_paths(self, mock_mlflow, mock_load_data,
                                          mock_split_train_test, mock_train, mock_validate):
        input_data_path = "/different/path/to/data.csv"

        mock_context_manager = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = mock_context_manager
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_load_data.return_value = "different_processed_data_path"
        mock_split_train_test.return_value = ("xtrain2", "xtest2", "ytrain2", "ytest2")
        mock_train.return_value = "model2_path"

        workflow(input_data_path)

        mock_load_data.assert_called_once_with(input_data_path)
        mock_split_train_test.assert_called_once_with("different_processed_data_path")
        mock_train.assert_called_once_with("xtrain2", "ytrain2")
        mock_validate.assert_called_once_with("model2_path", "xtest2", "ytest2")

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    def test_workflow_mlflow_context_manager(self, mock_mlflow, mock_load_data,
                                            mock_split_train_test, mock_train, mock_validate):
        input_data_path = "/path/to/data.csv"

        mock_context_manager = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_context_manager)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run

        mock_load_data.return_value = "data_path"
        mock_split_train_test.return_value = ("x1", "x2", "y1", "y2")
        mock_train.return_value = "model"

        workflow(input_data_path)

        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    def test_workflow_step_execution_order(self, mock_mlflow, mock_load_data,
                                          mock_split_train_test, mock_train, mock_validate):
        input_data_path = "/path/to/data.csv"

        mock_context_manager = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = mock_context_manager
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        execution_order = []

        def track_load_data(path):
            execution_order.append("load_data")
            return "data_path"

        def track_split(path):
            execution_order.append("split_train_test")
            return ("x1", "x2", "y1", "y2")

        def track_train(x_path, y_path):
            execution_order.append("train")
            return "model_path"

        def track_validate(model_path, x_test, y_test):
            execution_order.append("validate")

        mock_load_data.side_effect = track_load_data
        mock_split_train_test.side_effect = track_split
        mock_train.side_effect = track_train
        mock_validate.side_effect = track_validate

        workflow(input_data_path)

        assert execution_order == ["load_data", "split_train_test", "train", "validate"]

    def test_workflow_function_exists(self):
        assert callable(workflow)

    @patch('summit.training.main.validate')
    @patch('summit.training.main.train')
    @patch('summit.training.main.split_train_test')
    @patch('summit.training.main.load_data')
    @patch('summit.training.main.mlflow')
    def test_workflow_exception_handling_in_mlflow_context(self, mock_mlflow, mock_load_data,
                                                          mock_split_train_test, mock_train, mock_validate):
        input_data_path = "/path/to/data.csv"

        mock_load_data.side_effect = Exception("Load data failed")

        mock_context_manager = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = mock_context_manager
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(Exception, match="Load data failed"):
            workflow(input_data_path)

        mock_mlflow.start_run.assert_called_once()

