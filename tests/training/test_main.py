from unittest.mock import patch, Mock
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
