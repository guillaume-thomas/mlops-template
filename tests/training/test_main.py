import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import joblib
import pandas as pd
from summit.training.main import workflow


def test_workflow_runs_all_steps():
    with (
        patch("mlflow.start_run") as mock_run,
        patch("summit.training.main.load_data") as mock_load,
        patch("summit.training.main.split_train_test") as mock_split,
        patch("summit.training.main.train") as mock_train,
        patch("summit.training.main.validate"),
    ):
        mock_run.return_value.__enter__ = Mock()
        mock_run.return_value.__exit__ = Mock(return_value=False)

        mock_load.return_value = "data.csv"
        mock_split.return_value = ("x_train.csv", "x_test.csv", "y_train.csv", "y_test.csv")
        mock_train.return_value = "model.joblib"

        workflow("input.csv", n_estimators=10, max_depth=5, random_state=42)

        mock_load.assert_called_once()
        mock_split.assert_called_once()
        mock_train.assert_called_once()


def test_workflow_integration_with_real_data():  # ruff: noqa: PLR0915, C901
    data_path = Path("data/all_titanic.csv")
    assert data_path.exists(), f"Data file not found: {data_path}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_dir = Path(tmp_dir)

        mock_run_info = Mock()
        mock_run_info.info.run_id = "test-run-id"

        mock_model_info = Mock()
        mock_model_info.artifact_path = "model"
        mock_model_info.model_uri = "runs:/test-run-id/model"
        mock_model_info.model_uuid = "test-uuid"
        mock_model_info.metadata = {}

        with (
            patch("mlflow.start_run") as mock_start_run,
            patch("mlflow.active_run", return_value=mock_run_info),
            patch("mlflow.log_artifact") as mock_log_artifact,
            patch("mlflow.log_metric"),
            patch("mlflow.log_params"),
            patch("mlflow.log_dict"),
            patch("mlflow.sklearn.log_model", return_value=mock_model_info),
            patch("mlflow.register_model"),
            patch("boto3.client") as mock_s3_client,
        ):
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run_info)
            mock_start_run.return_value.__exit__ = Mock(return_value=False)

            mock_s3 = MagicMock()
            mock_s3_client.return_value = mock_s3

            def mock_download_file(bucket: str, key: str, local_path: str | Path) -> None:
                df = pd.read_csv(data_path)
                df.to_csv(local_path, index=False)

            mock_s3.download_file.side_effect = mock_download_file

            saved_artifacts = {}

            def mock_log_artifact_side_effect(local_file: str, artifact_path: str) -> None:
                local_path = Path(local_file)
                if local_path.exists():
                    artifact_key = f"{artifact_path}/{local_path.name}"
                    target_path = artifacts_dir / artifact_path / local_path.name
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    if local_path.suffix == ".csv":
                        df = pd.read_csv(local_path)
                        df.to_csv(target_path, index=False)
                    else:
                        shutil.copy2(local_path, target_path)

                    saved_artifacts[artifact_key] = str(target_path)

            mock_log_artifact.side_effect = mock_log_artifact_side_effect

            with patch("summit.training.steps.split_train_test.client.download_artifacts") as mock_split_download, \
                 patch("summit.training.steps.train.client.download_artifacts") as mock_train_download, \
                 patch("summit.training.steps.validate.client.download_artifacts") as mock_validate_download:

                def mock_download_artifacts_effect(run_id: str, path: str) -> str:
                    if path in saved_artifacts:
                        return saved_artifacts[path]
                    raise FileNotFoundError(f"Artifact {path} not found")

                mock_split_download.side_effect = mock_download_artifacts_effect
                mock_train_download.side_effect = mock_download_artifacts_effect
                mock_validate_download.side_effect = mock_download_artifacts_effect

                workflow("all_titanic.csv", n_estimators=10, max_depth=5, random_state=42)

            mock_s3.download_file.assert_called_once()
            assert mock_s3.download_file.call_args[0][0] == "summit"
            assert mock_s3.download_file.call_args[0][1] == "all_titanic.csv"

            assert "path_output/data.csv" in saved_artifacts
            assert "xtrain/xtrain.csv" in saved_artifacts
            assert "xtest/xtest.csv" in saved_artifacts
            assert "ytrain/ytrain.csv" in saved_artifacts
            assert "ytest/ytest.csv" in saved_artifacts
            assert "model_trained/model.joblib" in saved_artifacts

            xtrain_df = pd.read_csv(saved_artifacts["xtrain/xtrain.csv"])
            assert len(xtrain_df) > 0
            assert "Pclass" in xtrain_df.columns

            model_path = saved_artifacts["model_trained/model.joblib"]
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, "predict")

