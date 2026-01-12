from unittest.mock import patch
import pandas as pd

from summit.training.steps.train import train


def test_train_with_real_data(tmp_path):
    df = pd.read_csv("data/all_titanic.csv")

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_artifact"),
        patch("summit.training.steps.train.client") as mock_client,
    ):
        mock_run.return_value.info.run_id = "test-run"

        x_train = df[["Pclass", "Sex", "SibSp", "Parch"]].head(100)
        y_train = df[["Survived"]].head(100)

        x_file = tmp_path / "x_train.csv"
        y_file = tmp_path / "y_train.csv"
        x_train.to_csv(x_file, index=False)
        y_train.to_csv(y_file, index=False)

        mock_client.download_artifacts.side_effect = [str(x_file), str(y_file)]

        result = train("xtrain/xtrain.csv", "ytrain/ytrain.csv", n_estimators=10, max_depth=3, random_state=42)

        assert "model_trained" in result
        assert ".joblib" in result
