import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from summit.training.steps.train import train, ARTIFACT_PATH


class TestTrain:

    @patch('summit.training.steps.train.os.remove')
    @patch('summit.training.steps.train.mlflow.log_artifact')
    @patch('summit.training.steps.train.joblib.dump')
    @patch('summit.training.steps.train.linear_model.LinearRegression')
    @patch('summit.training.steps.train.pandas.read_csv')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.mlflow')
    def test_train_success(self, mock_mlflow, mock_client, mock_read_csv,
                          mock_linear_regression, mock_joblib_dump, mock_log_artifact, mock_remove):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.side_effect = [
            "/downloaded/xtrain.csv",
            "/downloaded/ytrain.csv"
        ]

        mock_x_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        mock_y_train = pd.DataFrame({
            'target': [0.5, 1.5, 2.5]
        })

        mock_read_csv.side_effect = [mock_x_train, mock_y_train]

        mock_model = MagicMock()
        mock_linear_regression.return_value = mock_model

        result = train(x_train_path, y_train_path)

        assert mock_client.download_artifacts.call_count == 2
        mock_client.download_artifacts.assert_any_call(
            run_id="test_run_id",
            path=x_train_path
        )
        mock_client.download_artifacts.assert_any_call(
            run_id="test_run_id",
            path=y_train_path
        )

        assert mock_read_csv.call_count == 2
        mock_read_csv.assert_any_call("/downloaded/xtrain.csv", index_col=False)
        mock_read_csv.assert_any_call("/downloaded/ytrain.csv", index_col=False)

        mock_linear_regression.assert_called_once()
        mock_model.fit.assert_called_once_with(mock_x_train, mock_y_train)

        mock_joblib_dump.assert_called_once_with(mock_model, "./model.joblib")
        mock_log_artifact.assert_called_once_with("./model.joblib", ARTIFACT_PATH)
        mock_remove.assert_called_once_with("./model.joblib")

        assert result == f"{ARTIFACT_PATH}/model.joblib"

    @patch('summit.training.steps.train.logging')
    @patch('summit.training.steps.train.mlflow')
    @patch('summit.training.steps.train.client')
    def test_train_logging(self, mock_client, mock_mlflow, mock_logging):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        with patch('summit.training.steps.train.pandas.read_csv'), \
             patch('summit.training.steps.train.linear_model.LinearRegression'), \
             patch('summit.training.steps.train.joblib.dump'), \
             patch('summit.training.steps.train.os.remove'):

            train(x_train_path, y_train_path)

            mock_logging.warning.assert_called_with(f"train {x_train_path} {y_train_path}")

    @patch('summit.training.steps.train.os.remove')
    @patch('summit.training.steps.train.mlflow.log_artifact')
    @patch('summit.training.steps.train.joblib.dump')
    @patch('summit.training.steps.train.linear_model.LinearRegression')
    @patch('summit.training.steps.train.pandas.read_csv')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.mlflow')
    def test_train_model_fitting(self, mock_mlflow, mock_client, mock_read_csv,
                                mock_linear_regression, mock_joblib_dump, mock_log_artifact, mock_remove):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_x_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        mock_y_train = pd.DataFrame({
            'target': [1.0, 2.0, 3.0, 4.0]
        })

        mock_read_csv.side_effect = [mock_x_train, mock_y_train]

        mock_model = MagicMock(spec=['fit', 'predict'])
        mock_linear_regression.return_value = mock_model

        train(x_train_path, y_train_path)

        mock_model.fit.assert_called_once_with(mock_x_train, mock_y_train)

    @patch('summit.training.steps.train.os.remove')
    @patch('summit.training.steps.train.mlflow.log_artifact')
    @patch('summit.training.steps.train.joblib.dump')
    @patch('summit.training.steps.train.linear_model.LinearRegression')
    @patch('summit.training.steps.train.pandas.read_csv')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.mlflow')
    def test_train_file_operations(self, mock_mlflow, mock_client, mock_read_csv,
                                  mock_linear_regression, mock_joblib_dump, mock_log_artifact, mock_remove):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_linear_regression.return_value = mock_model

        train(x_train_path, y_train_path)

        model_filename = "model.joblib"
        model_path = "./" + model_filename

        mock_joblib_dump.assert_called_once_with(mock_model, model_path)
        mock_log_artifact.assert_called_once_with(model_path, ARTIFACT_PATH)
        mock_remove.assert_called_once_with(model_path)

    @patch('summit.training.steps.train.mlflow')
    @patch('summit.training.steps.train.client')
    def test_train_mlflow_client_usage(self, mock_client, mock_mlflow):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        with patch('summit.training.steps.train.pandas.read_csv'), \
             patch('summit.training.steps.train.linear_model.LinearRegression'), \
             patch('summit.training.steps.train.joblib.dump'), \
             patch('summit.training.steps.train.os.remove'):

            train(x_train_path, y_train_path)

            assert mock_client.download_artifacts.call_count == 2
            mock_client.download_artifacts.assert_any_call(
                run_id="test_run_id",
                path=x_train_path
            )
            mock_client.download_artifacts.assert_any_call(
                run_id="test_run_id",
                path=y_train_path
            )

    def test_train_artifact_path_constant(self):
        assert ARTIFACT_PATH == "model_trained"

    def test_train_function_exists(self):
        assert callable(train)

    @patch('summit.training.steps.train.mlflow')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.pandas.read_csv')
    def test_train_data_loading_failure(self, mock_read_csv, mock_client, mock_mlflow):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_read_csv.side_effect = Exception("Failed to read CSV")

        with pytest.raises(Exception, match="Failed to read CSV"):
            train(x_train_path, y_train_path)

    @patch('summit.training.steps.train.os.remove')
    @patch('summit.training.steps.train.mlflow.log_artifact')
    @patch('summit.training.steps.train.joblib.dump')
    @patch('summit.training.steps.train.linear_model.LinearRegression')
    @patch('summit.training.steps.train.pandas.read_csv')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.mlflow')
    def test_train_model_dump_failure(self, mock_mlflow, mock_client, mock_read_csv,
                                     mock_linear_regression, mock_joblib_dump, mock_log_artifact, mock_remove):
        x_train_path = "xtrain/xtrain.csv"
        y_train_path = "ytrain/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_joblib_dump.side_effect = Exception("Failed to dump model")

        with pytest.raises(Exception, match="Failed to dump model"):
            train(x_train_path, y_train_path)

    @patch('summit.training.steps.train.os.remove')
    @patch('summit.training.steps.train.mlflow.log_artifact')
    @patch('summit.training.steps.train.joblib.dump')
    @patch('summit.training.steps.train.linear_model.LinearRegression')
    @patch('summit.training.steps.train.pandas.read_csv')
    @patch('summit.training.steps.train.client')
    @patch('summit.training.steps.train.mlflow')
    def test_train_with_different_paths(self, mock_mlflow, mock_client, mock_read_csv,
                                       mock_linear_regression, mock_joblib_dump, mock_log_artifact, mock_remove):
        x_train_path = "different/xtrain.csv"
        y_train_path = "different/ytrain.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "different_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_linear_regression.return_value = mock_model

        result = train(x_train_path, y_train_path)

        mock_client.download_artifacts.assert_any_call(
            run_id="different_run_id",
            path=x_train_path
        )
        mock_client.download_artifacts.assert_any_call(
            run_id="different_run_id",
            path=y_train_path
        )

        assert result == f"{ARTIFACT_PATH}/model.joblib"
