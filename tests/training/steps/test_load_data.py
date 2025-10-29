import pytest
from unittest.mock import patch, MagicMock

from summit.training.steps.load_data import load_data, ARTIFACT_PATH, LOCAL_PATH


class TestLoadData:

    @patch('summit.training.steps.load_data.os.remove')
    @patch('summit.training.steps.load_data.mlflow')
    @patch('summit.training.steps.load_data.Path')
    @patch('summit.training.steps.load_data.boto3')
    @patch('summit.training.steps.load_data.os.environ.get')
    @patch('summit.training.steps.load_data.logging')
    def test_load_data_success(self, mock_logging, mock_env_get, mock_boto3,
                              mock_path, mock_mlflow, mock_remove):
        path = "test/data.csv"

        mock_env_get.side_effect = lambda key: {
            "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret"
        }.get(key)

        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        mock_path_instance = MagicMock()
        mock_path_instance.name = "data.csv"
        mock_path.return_value = mock_path_instance

        result = load_data(path)

        mock_logging.warning.assert_called_with(f"load_data on path : {path}")
        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        mock_s3_client.download_file.assert_called_once_with("summit", path, LOCAL_PATH)
        mock_path.assert_called_once_with(LOCAL_PATH)
        mock_mlflow.log_artifact.assert_called_once_with("data.csv", ARTIFACT_PATH)
        mock_remove.assert_called_once_with(LOCAL_PATH)
        assert result == f"{ARTIFACT_PATH}/data.csv"

    @patch('summit.training.steps.load_data.os.remove')
    @patch('summit.training.steps.load_data.mlflow')
    @patch('summit.training.steps.load_data.Path')
    @patch('summit.training.steps.load_data.boto3')
    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_with_different_filename(self, mock_env_get, mock_boto3,
                                              mock_path, mock_mlflow, mock_remove):
        path = "different/file.csv"

        mock_env_get.side_effect = lambda key: {
            "MLFLOW_S3_ENDPOINT_URL": "http://test:9000",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret"
        }.get(key)

        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        mock_path_instance = MagicMock()
        mock_path_instance.name = "file.csv"
        mock_path.return_value = mock_path_instance

        result = load_data(path)

        mock_s3_client.download_file.assert_called_once_with("summit", path, LOCAL_PATH)
        mock_mlflow.log_artifact.assert_called_once_with("file.csv", ARTIFACT_PATH)
        assert result == f"{ARTIFACT_PATH}/file.csv"

    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_env_variables_configuration(self, mock_env_get):
        mock_env_get.side_effect = lambda key: {
            "MLFLOW_S3_ENDPOINT_URL": "http://custom:9000",
            "AWS_ACCESS_KEY_ID": "custom_key",
            "AWS_SECRET_ACCESS_KEY": "custom_secret"
        }.get(key)

        with patch('summit.training.steps.load_data.boto3') as mock_boto3:
            mock_s3_client = MagicMock()
            mock_boto3.client.return_value = mock_s3_client

            with patch('summit.training.steps.load_data.mlflow'), \
                 patch('summit.training.steps.load_data.Path'), \
                 patch('summit.training.steps.load_data.os.remove'):

                load_data("test.csv")

                mock_boto3.client.assert_called_once_with(
                    "s3",
                    endpoint_url="http://custom:9000",
                    aws_access_key_id="custom_key",
                    aws_secret_access_key="custom_secret"
                )

    @patch('summit.training.steps.load_data.boto3')
    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_s3_download_failure(self, mock_env_get, mock_boto3):
        path = "test/data.csv"

        mock_env_get.return_value = "test_value"

        mock_s3_client = MagicMock()
        mock_s3_client.download_file.side_effect = Exception("S3 download failed")
        mock_boto3.client.return_value = mock_s3_client

        with pytest.raises(Exception, match="S3 download failed"):
            load_data(path)

    @patch('summit.training.steps.load_data.os.remove')
    @patch('summit.training.steps.load_data.mlflow')
    @patch('summit.training.steps.load_data.Path')
    @patch('summit.training.steps.load_data.boto3')
    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_mlflow_log_artifact_failure(self, mock_env_get, mock_boto3,
                                                   mock_path, mock_mlflow, mock_remove):
        path = "test/data.csv"

        mock_env_get.return_value = "test_value"
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        mock_path_instance = MagicMock()
        mock_path_instance.name = "data.csv"
        mock_path.return_value = mock_path_instance

        mock_mlflow.log_artifact.side_effect = Exception("MLflow logging failed")

        with pytest.raises(Exception, match="MLflow logging failed"):
            load_data(path)

    def test_load_data_constants(self):
        assert ARTIFACT_PATH == "path_output"
        assert LOCAL_PATH == "./data.csv"

    def test_load_data_function_exists(self):
        assert callable(load_data)

    @patch('summit.training.steps.load_data.os.remove')
    @patch('summit.training.steps.load_data.mlflow')
    @patch('summit.training.steps.load_data.Path')
    @patch('summit.training.steps.load_data.boto3')
    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_file_cleanup(self, mock_env_get, mock_boto3,
                                   mock_path, mock_mlflow, mock_remove):
        path = "test/data.csv"

        mock_env_get.return_value = "test_value"
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        mock_path_instance = MagicMock()
        mock_path_instance.name = "data.csv"
        mock_path.return_value = mock_path_instance

        load_data(path)

        mock_remove.assert_called_once_with(LOCAL_PATH)

    @patch('summit.training.steps.load_data.os.environ.get')
    def test_load_data_env_variables_none_values(self, mock_env_get):
        mock_env_get.return_value = None

        with patch('summit.training.steps.load_data.boto3') as mock_boto3:
            mock_s3_client = MagicMock()
            mock_boto3.client.return_value = mock_s3_client

            with patch('summit.training.steps.load_data.mlflow'), \
                 patch('summit.training.steps.load_data.Path'), \
                 patch('summit.training.steps.load_data.os.remove'):

                load_data("test.csv")

                mock_boto3.client.assert_called_once_with(
                    "s3",
                    endpoint_url=None,
                    aws_access_key_id=None,
                    aws_secret_access_key=None
                )
