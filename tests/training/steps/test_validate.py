import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from summit.training.steps.validate import validate


class TestValidate:

    @patch('summit.training.steps.validate.mlflow.register_model')
    @patch('summit.training.steps.validate.mlflow.sklearn.log_model')
    @patch('summit.training.steps.validate.mlflow.log_dict')
    @patch('summit.training.steps.validate.mlflow.log_metric')
    @patch('summit.training.steps.validate.infer_signature')
    @patch('summit.training.steps.validate.median_absolute_error')
    @patch('summit.training.steps.validate.r2_score')
    @patch('summit.training.steps.validate.mean_absolute_error')
    @patch('summit.training.steps.validate.mean_squared_error')
    @patch('summit.training.steps.validate.pd.read_csv')
    @patch('summit.training.steps.validate.joblib.load')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.mlflow')
    def test_validate_success(self, mock_mlflow, mock_client, mock_joblib_load, mock_read_csv,
                             mock_mse, mock_mae, mock_r2, mock_medae, mock_infer_signature,
                             mock_log_metric, mock_log_dict, mock_log_model, mock_register_model):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.side_effect = [
            "/downloaded/model.joblib",
            "/downloaded/xtest.csv",
            "/downloaded/ytest.csv"
        ]

        mock_model = MagicMock(spec=['coef_', 'predict'])
        mock_model.coef_ = np.array([0.5, 1.2, -0.3])
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        mock_joblib_load.return_value = mock_model

        mock_x_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30],
            'feature3': [0.1, 0.2, 0.3]
        })
        mock_y_test_df = pd.DataFrame({'target': [1.1, 2.1, 3.1]})

        mock_read_csv.side_effect = [mock_x_test, mock_y_test_df]

        mock_mse.return_value = 0.01
        mock_mae.return_value = 0.05
        mock_r2.return_value = 0.95
        mock_medae.return_value = 0.03

        mock_signature = MagicMock()
        mock_infer_signature.return_value = mock_signature

        mock_model_info = MagicMock()
        mock_model_info.artifact_path = "model_final"
        mock_model_info.model_uri = "runs:/test_run_id/model_final"
        mock_model_info.model_uuid = "uuid-123"
        mock_model_info.metadata = {"key": "value"}
        mock_log_model.return_value = mock_model_info

        validate(model_path, x_test_path, y_test_path)

        assert mock_client.download_artifacts.call_count == 3
        mock_client.download_artifacts.assert_any_call(run_id="test_run_id", path=model_path)
        mock_client.download_artifacts.assert_any_call(run_id="test_run_id", path=x_test_path)
        mock_client.download_artifacts.assert_any_call(run_id="test_run_id", path=y_test_path)

        mock_joblib_load.assert_called_once_with("/downloaded/model.joblib")

        assert mock_read_csv.call_count == 2
        mock_read_csv.assert_any_call("/downloaded/xtest.csv", index_col=False)
        mock_read_csv.assert_any_call("/downloaded/ytest.csv", index_col=False)

        expected_y_test = pd.Series([1.1, 2.1, 3.1])
        mock_model.predict.assert_called_once()

        mock_mse.assert_called_once()
        mock_mae.assert_called_once()
        mock_r2.assert_called_once()
        mock_medae.assert_called_once()

        mock_log_metric.assert_any_call("mse", 0.01)
        mock_log_metric.assert_any_call("mae", 0.05)
        mock_log_metric.assert_any_call("r2", 0.95)
        mock_log_metric.assert_any_call("medae", 0.03)

        expected_feature_importance = {
            'feature1': 0.5,
            'feature2': 1.2,
            'feature3': -0.3
        }
        mock_log_dict.assert_called_once_with(expected_feature_importance, "feature_importance.json")

        mock_log_model.assert_called_once()
        mock_register_model.assert_called_once_with(mock_model_info.model_uri, "model_registered")

    @patch('summit.training.steps.validate.logging')
    @patch('summit.training.steps.validate.mlflow')
    @patch('summit.training.steps.validate.client')
    def test_validate_logging(self, mock_client, mock_mlflow, mock_logging):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        with patch('summit.training.steps.validate.joblib.load'), \
             patch('summit.training.steps.validate.pd.read_csv'), \
             patch('summit.training.steps.validate.mean_squared_error'), \
             patch('summit.training.steps.validate.mean_absolute_error'), \
             patch('summit.training.steps.validate.r2_score'), \
             patch('summit.training.steps.validate.median_absolute_error'), \
             patch('summit.training.steps.validate.mlflow.log_metric'), \
             patch('summit.training.steps.validate.mlflow.log_dict'), \
             patch('summit.training.steps.validate.mlflow.sklearn.log_model'), \
             patch('summit.training.steps.validate.mlflow.register_model'):

            validate(model_path, x_test_path, y_test_path)

            mock_logging.warning.assert_any_call(f"validate {model_path}")

    @patch('summit.training.steps.validate.mlflow.register_model')
    @patch('summit.training.steps.validate.mlflow.sklearn.log_model')
    @patch('summit.training.steps.validate.mlflow.log_dict')
    @patch('summit.training.steps.validate.mlflow.log_metric')
    @patch('summit.training.steps.validate.infer_signature')
    @patch('summit.training.steps.validate.median_absolute_error')
    @patch('summit.training.steps.validate.r2_score')
    @patch('summit.training.steps.validate.mean_absolute_error')
    @patch('summit.training.steps.validate.mean_squared_error')
    @patch('summit.training.steps.validate.pd.read_csv')
    @patch('summit.training.steps.validate.joblib.load')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.mlflow')
    def test_validate_y_test_single_column_handling(self, mock_mlflow, mock_client, mock_joblib_load, mock_read_csv,
                                                   mock_mse, mock_mae, mock_r2, mock_medae, mock_infer_signature,
                                                   mock_log_metric, mock_log_dict, mock_log_model, mock_register_model):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.coef_ = np.array([0.5])
        mock_model.predict.return_value = np.array([1.0])
        mock_joblib_load.return_value = mock_model

        mock_x_test = pd.DataFrame({'feature1': [1]})
        mock_y_test_df = pd.DataFrame({'target': [1.1]})

        mock_read_csv.side_effect = [mock_x_test, mock_y_test_df]

        validate(model_path, x_test_path, y_test_path)

        assert mock_y_test_df.shape[1] == 1

    @patch('summit.training.steps.validate.mlflow.register_model')
    @patch('summit.training.steps.validate.mlflow.sklearn.log_model')
    @patch('summit.training.steps.validate.mlflow.log_dict')
    @patch('summit.training.steps.validate.mlflow.log_metric')
    @patch('summit.training.steps.validate.infer_signature')
    @patch('summit.training.steps.validate.median_absolute_error')
    @patch('summit.training.steps.validate.r2_score')
    @patch('summit.training.steps.validate.mean_absolute_error')
    @patch('summit.training.steps.validate.mean_squared_error')
    @patch('summit.training.steps.validate.pd.read_csv')
    @patch('summit.training.steps.validate.joblib.load')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.mlflow')
    def test_validate_feature_importance_multidimensional_coefs(self, mock_mlflow, mock_client, mock_joblib_load, mock_read_csv,
                                                               mock_mse, mock_mae, mock_r2, mock_medae, mock_infer_signature,
                                                               mock_log_metric, mock_log_dict, mock_log_model, mock_register_model):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, 1.2, -0.3]])
        mock_model.predict.return_value = np.array([1.0])
        mock_joblib_load.return_value = mock_model

        mock_x_test = pd.DataFrame({
            'feature1': [1],
            'feature2': [2],
            'feature3': [3]
        })
        mock_y_test_df = pd.DataFrame({'target': [1.1]})

        mock_read_csv.side_effect = [mock_x_test, mock_y_test_df]

        validate(model_path, x_test_path, y_test_path)

        expected_feature_importance = {
            'feature1': 0.5,
            'feature2': 1.2,
            'feature3': -0.3
        }
        mock_log_dict.assert_called_once_with(expected_feature_importance, "feature_importance.json")

    @patch('summit.training.steps.validate.logging')
    @patch('summit.training.steps.validate.mlflow.register_model')
    @patch('summit.training.steps.validate.mlflow.sklearn.log_model')
    @patch('summit.training.steps.validate.mlflow.log_dict')
    @patch('summit.training.steps.validate.mlflow.log_metric')
    @patch('summit.training.steps.validate.infer_signature')
    @patch('summit.training.steps.validate.median_absolute_error')
    @patch('summit.training.steps.validate.r2_score')
    @patch('summit.training.steps.validate.mean_absolute_error')
    @patch('summit.training.steps.validate.mean_squared_error')
    @patch('summit.training.steps.validate.pd.read_csv')
    @patch('summit.training.steps.validate.joblib.load')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.mlflow')
    def test_validate_model_registration_failure(self, mock_mlflow, mock_client, mock_joblib_load, mock_read_csv,
                                                 mock_mse, mock_mae, mock_r2, mock_medae, mock_infer_signature,
                                                 mock_log_metric, mock_log_dict, mock_log_model, mock_register_model,
                                                 mock_logging):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.coef_ = np.array([0.5])
        mock_model.predict.return_value = np.array([1.0])
        mock_joblib_load.return_value = mock_model

        mock_x_test = pd.DataFrame({'feature1': [1]})
        mock_y_test_df = pd.DataFrame({'target': [1.1]})

        mock_read_csv.side_effect = [mock_x_test, mock_y_test_df]

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/test_run_id/model_final"
        mock_log_model.return_value = mock_model_info

        mock_register_model.side_effect = Exception("Registration failed")

        validate(model_path, x_test_path, y_test_path)

        mock_logging.error.assert_called_with("Erreur registry: Registration failed")

    @patch('summit.training.steps.validate.mlflow.sklearn.log_model')
    @patch('summit.training.steps.validate.mlflow.log_dict')
    @patch('summit.training.steps.validate.mlflow.log_metric')
    @patch('summit.training.steps.validate.infer_signature')
    @patch('summit.training.steps.validate.median_absolute_error')
    @patch('summit.training.steps.validate.r2_score')
    @patch('summit.training.steps.validate.mean_absolute_error')
    @patch('summit.training.steps.validate.mean_squared_error')
    @patch('summit.training.steps.validate.pd.read_csv')
    @patch('summit.training.steps.validate.joblib.load')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.mlflow')
    def test_validate_mlflow_model_logging(self, mock_mlflow, mock_client, mock_joblib_load, mock_read_csv,
                                          mock_mse, mock_mae, mock_r2, mock_medae, mock_infer_signature,
                                          mock_log_metric, mock_log_dict, mock_log_model):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.coef_ = np.array([0.5])
        mock_model.predict.return_value = np.array([1.0])
        mock_joblib_load.return_value = mock_model

        mock_x_test = pd.DataFrame({'feature1': [1]})
        mock_y_test_df = pd.DataFrame({'target': [1.1]})

        mock_read_csv.side_effect = [mock_x_test, mock_y_test_df]

        mock_signature = MagicMock()
        mock_infer_signature.return_value = mock_signature

        with patch('summit.training.steps.validate.mlflow.register_model'):
            validate(model_path, x_test_path, y_test_path)

            mock_log_model.assert_called_once()
            args, kwargs = mock_log_model.call_args
            assert kwargs['name'] == "model_final"
            assert kwargs['signature'] == mock_signature

    def test_validate_function_exists(self):
        assert callable(validate)

    @patch('summit.training.steps.validate.mlflow')
    @patch('summit.training.steps.validate.client')
    @patch('summit.training.steps.validate.joblib.load')
    def test_validate_model_loading_failure(self, mock_joblib_load, mock_client, mock_mlflow):
        model_path = "model_trained/model.joblib"
        x_test_path = "xtest/xtest.csv"
        y_test_path = "ytest/ytest.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_joblib_load.side_effect = Exception("Failed to load model")

        with pytest.raises(Exception, match="Failed to load model"):
            validate(model_path, x_test_path, y_test_path)
