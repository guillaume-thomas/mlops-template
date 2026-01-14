from unittest.mock import patch
import shutil
import pandas as pd

from summit.training.steps.split_train_test import split_train_test, FEATURES, TARGET


def test_split_train_test_with_real_data(tmp_path):
    """Test que split_train_test sépare correctement les données."""
    data_file = "data/all_titanic.csv"
    original_df = pd.read_csv(data_file)
    original_size = len(original_df)

    saved_files = {}

    def mock_log_artifact_side_effect(path, artifact_path):
        """Capture les fichiers splits avant suppression par TemporaryDirectory."""
        filename = artifact_path.split("/")[-1] if "/" in artifact_path else artifact_path
        saved_path = tmp_path / f"saved_{filename}.csv"
        shutil.copy(path, saved_path)
        saved_files[artifact_path] = saved_path

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_artifact", side_effect=mock_log_artifact_side_effect),
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

        xtrain = pd.read_csv(saved_files["xtrain"])
        xtest = pd.read_csv(saved_files["xtest"])
        ytrain = pd.read_csv(saved_files["ytrain"])
        ytest = pd.read_csv(saved_files["ytest"])

        assert list(xtrain.columns) == FEATURES, "X_train doit contenir les features"
        assert list(xtest.columns) == FEATURES, "X_test doit contenir les features"
        assert list(ytrain.columns) == [TARGET], "y_train doit contenir la target"
        assert list(ytest.columns) == [TARGET], "y_test doit contenir la target"

        assert len(xtrain) == len(ytrain), "X_train et y_train doivent avoir la même taille"
        assert len(xtest) == len(ytest), "X_test et y_test doivent avoir la même taille"

        total_split_size = len(xtrain) + len(xtest)
        assert total_split_size == original_size, (
            f"La somme des splits ({total_split_size}) doit égaler la taille originale ({original_size})"
        )

        test_ratio = len(xtest) / total_split_size
        assert 0.25 < test_ratio < 0.35, f"Le ratio test ({test_ratio:.2f}) devrait être proche de 0.3"
