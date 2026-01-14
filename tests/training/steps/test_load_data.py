from unittest.mock import patch, Mock
import shutil
import pandas as pd

from summit.training.steps.load_data import load_data


def test_load_data_with_local_file(tmp_path):
    """Test que load_data télécharge, profile et log correctement les données."""
    data_file = "data/all_titanic.csv"
    original_df = pd.read_csv(data_file)
    saved_csv_path = None

    def mock_log_artifact_side_effect(path, artifact_path):
        """Capture le fichier CSV avant qu'il ne soit supprimé par TemporaryDirectory.

        load_data() utilise un TemporaryDirectory qui supprime les fichiers à la fin.
        Ce side_effect copie le CSV loggé pour pouvoir le comparer après coup.
        """
        nonlocal saved_csv_path
        if path.endswith(".csv"):
            saved_csv_path = tmp_path / "saved_data.csv"
            shutil.copy(path, saved_csv_path)

    with patch("mlflow.log_artifact", side_effect=mock_log_artifact_side_effect), patch("boto3.client") as mock_s3:
        mock_client = Mock()

        def fake_download(bucket, key, local_path):
            shutil.copy(data_file, local_path)

        mock_client.download_file = fake_download
        mock_s3.return_value = mock_client

        result = load_data("all_titanic.csv")

        assert "path_output" in result
        assert ".csv" in result

        assert saved_csv_path is not None, "Le fichier CSV devrait avoir été loggé"
        assert saved_csv_path.exists(), "Le fichier CSV sauvegardé devrait exister"

        logged_df = pd.read_csv(saved_csv_path)
        pd.testing.assert_frame_equal(original_df, logged_df, check_dtype=False)
