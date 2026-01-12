from unittest.mock import patch, Mock
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from summit.training.steps.validate import validate


def test_validate_with_real_model_and_data(tmp_path):
    df = pd.read_csv("data/all_titanic.csv")

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_metric"),
        patch("mlflow.log_dict"),
        patch("mlflow.sklearn.log_model") as mock_log_model,
        patch("mlflow.register_model"),
        patch("summit.training.steps.validate.client") as mock_client,
    ):
        mock_run.return_value.info.run_id = "test-run"
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/test/model"
        mock_log_model.return_value = mock_model_info

        x_test = df[["Pclass", "Sex", "SibSp", "Parch"]].head(50)
        y_test = df[["Survived"]].head(50)

        x_dummies = pd.get_dummies(x_test)
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(x_dummies, y_test.iloc[:, 0])

        model_file = tmp_path / "model.joblib"
        x_file = tmp_path / "x_test.csv"
        y_file = tmp_path / "y_test.csv"

        joblib.dump(model, model_file)
        x_test.to_csv(x_file, index=False)
        y_test.to_csv(y_file, index=False)

        mock_client.download_artifacts.side_effect = [str(model_file), str(x_file), str(y_file)]

        validate("model_trained/model.joblib", "xtest/xtest.csv", "ytest/ytest.csv")
