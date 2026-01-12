from unittest.mock import patch, Mock
import shutil

from summit.training.steps.load_data import load_data


def test_load_data_with_local_file(tmp_path):
    data_file = "data/all_titanic.csv"

    with patch("mlflow.log_artifact"), patch("boto3.client") as mock_s3:
        mock_client = Mock()

        def fake_download(bucket, key, local_path):
            shutil.copy(data_file, local_path)

        mock_client.download_file = fake_download
        mock_s3.return_value = mock_client

        result = load_data("all_titanic.csv")

        assert "path_output" in result
        assert ".csv" in result
