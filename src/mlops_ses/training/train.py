import os

import fire
import joblib
import mlflow.sklearn
import pandas
from sklearn import linear_model


def train(x_train_path, y_train_path):
    print(f"train {x_train_path} {y_train_path}")
    x_train = pandas.read_csv(x_train_path)
    y_train = pandas.read_csv(y_train_path)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    model_path = "./model.joblib"

    #joblib.dump(model, model_path, protocol=5)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")

    os.remove(model_path)



if __name__ == "__main__":
    fire.Fire(train)