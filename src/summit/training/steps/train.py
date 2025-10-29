import logging
import os

import fire
import joblib
import mlflow.sklearn
import pandas
from sklearn.ensemble import RandomForestClassifier

client = mlflow.MlflowClient()

ARTIFACT_PATH = "model_trained"

def train(x_train_path: str, y_train_path: str, n_estimators: int, max_depth: int, random_state: int) -> str:
    logging.warning(f"train {x_train_path} {y_train_path}")
    x_train = pandas.read_csv(client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=x_train_path), index_col=False)
    y_train = pandas.read_csv(client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=y_train_path), index_col=False)

    x_train = pandas.get_dummies(x_train)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)

    model_filename = "model.joblib"
    model_path = "./" + model_filename

    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, ARTIFACT_PATH)

    os.remove(model_path)

    return f"{ARTIFACT_PATH}/{model_filename}"



if __name__ == "__main__":
    fire.Fire(train)