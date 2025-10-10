import os

import fire
import pandas as pd
import sklearn.model_selection
import mlflow


def split_train_test(output_path):
    print(f"split on {output_path}")

    df = pd.read_csv(output_path)

    y = df["target"]
    x = df.drop(columns="target")
    x_train, x_test, y_train, y_test = (
        sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=42))

    path_xtrain = "./xtrain.csv"
    path_xtest = "./xtest.csv"
    path_ytrain = "./ytrain.csv"
    path_ytest = "./ytest.csv"

    x_train.to_csv(path_xtrain)
    x_test.to_csv(path_xtest)
    y_train.to_csv(path_ytrain)
    y_test.to_csv(path_ytest)

    mlflow.log_artifact(path_xtrain, "xtrain")
    mlflow.log_artifact(path_xtest, "xtest")
    mlflow.log_artifact(path_ytrain, "ytrain")
    mlflow.log_artifact(path_ytest, "ytest")

    os.remove(path_xtrain)
    os.remove(path_xtest)
    os.remove(path_ytrain)
    os.remove(path_ytest)

if __name__ == "__main__":
    fire.Fire(split_train_test)