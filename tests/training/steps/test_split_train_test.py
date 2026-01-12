from unittest.mock import patch
import shutil

from summit.training.steps.split_train_test import split_train_test


def test_split_train_test_with_real_data(tmp_path):
    data_file = "data/all_titanic.csv"

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_artifact"),
        patch("summit.training.steps.split_train_test.client") as mock_client,
    ):
        mock_run.return_value.info.run_id = "test-run"

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        data_copy = artifacts_dir / "data.csv"
        shutil.copy(data_file, data_copy)

        mock_client.download_artifacts.return_value = str(data_copy)

        result = split_train_test("path_output/data.csv")

        assert len(result) == 4
        assert all(".csv" in path for path in result)
        assert "xtrain" in result[0]
        assert "xtest" in result[1]
        assert "ytrain" in result[2]
        assert "ytest" in result[3]
