from unittest.mock import patch
import pandas as pd
import joblib
import shutil
import numpy as np

from summit.training.steps.train import train


def test_train_with_real_data(tmp_path):
    """Test que train entraîne un modèle fonctionnel."""
    df = pd.read_csv("data/all_titanic.csv")
    saved_model_path = None

    def mock_log_artifact_side_effect(path, artifact_path):
        """Capture le modèle avant suppression par TemporaryDirectory."""
        nonlocal saved_model_path
        if path.endswith(".joblib"):
            saved_model_path = tmp_path / "saved_model.joblib"
            shutil.copy(path, saved_model_path)

    with (
        patch("mlflow.active_run") as mock_run,
        patch("mlflow.log_artifact", side_effect=mock_log_artifact_side_effect),
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

        assert saved_model_path is not None, "Le modèle devrait avoir été sauvegardé"
        assert saved_model_path.exists(), "Le fichier modèle devrait exister"

        model = joblib.load(saved_model_path)
        assert model is not None, "Le modèle devrait être chargeable"

        x_test = pd.get_dummies(x_train.head(5))
        predictions = model.predict(x_test)

        assert len(predictions) == 5, "Le modèle devrait prédire pour 5 échantillons"
        assert all(pred in [0, 1] for pred in predictions), "Les prédictions devraient être 0 ou 1"
        assert hasattr(model, "predict_proba"), "Le modèle devrait avoir une méthode predict_proba"

        probas = model.predict_proba(x_test)
        assert probas.shape == (5, 2), "Les probabilités devraient être de shape (5, 2)"
        assert np.allclose(probas.sum(axis=1), 1.0), "Les probabilités devraient sommer à 1"
