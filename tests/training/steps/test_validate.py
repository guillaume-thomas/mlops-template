from unittest.mock import patch, Mock
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from summit.training.steps.validate import validate


def test_validate_with_real_model_and_data(tmp_path):
    """Test que validate calcule et log des métriques valides."""
    df = pd.read_csv("data/all_titanic.csv")
    logged_metrics = {}
    logged_dicts = {}

    def capture_metric(key, value):
        """Capture les métriques loggées pour vérification."""
        logged_metrics[key] = value

    def capture_dict(data, artifact_file):
        """Capture les dictionnaires loggés pour vérification."""
        logged_dicts[artifact_file] = data

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_metric", side_effect=capture_metric),
        patch("mlflow.log_dict", side_effect=capture_dict),
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

        assert "mse" in logged_metrics, "MSE devrait être loggé"
        assert "mae" in logged_metrics, "MAE devrait être loggé"
        assert "r2" in logged_metrics, "R2 devrait être loggé"
        assert "medae" in logged_metrics, "MedAE devrait être loggé"

        assert logged_metrics["mse"] >= 0, "MSE devrait être positif"
        assert logged_metrics["mae"] >= 0, "MAE devrait être positif"
        assert -1 <= logged_metrics["r2"] <= 1, "R2 devrait être entre -1 et 1"
        assert logged_metrics["medae"] >= 0, "MedAE devrait être positif"

        assert logged_metrics["mae"] <= logged_metrics["mse"], "MAE devrait être <= MSE (en général)"

        assert "feature_importance.json" in logged_dicts, "Feature importance devrait être loggé"
        feature_importance = logged_dicts["feature_importance.json"]
        assert len(feature_importance) > 0, "Feature importance ne devrait pas être vide"
        assert all(isinstance(v, (int, float)) for v in feature_importance.values()), (
            "Les importances devraient être numériques"
        )

        mock_log_model.assert_called_once()
        call_kwargs = mock_log_model.call_args.kwargs
        assert "signature" in call_kwargs, "Le modèle devrait être loggé avec une signature"
        assert "input_example" in call_kwargs, "Le modèle devrait être loggé avec un input_example"
