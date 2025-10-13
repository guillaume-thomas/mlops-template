import logging

import fire
import mlflow

from summit.training.steps.load_data import load_data
from summit.training.steps.validate import validate
from summit.training.steps.split_train_test import split_train_test
from summit.training.steps.train import train


def workflow(input_data_path: str):
    logging.warning(f"workflow input path : {input_data_path}")
    with mlflow.start_run() :
        data_path = load_data(input_data_path)
        xtrain_path, xtest_path, ytrain_path, ytest_path = split_train_test(data_path)
        model_path = train(xtrain_path, ytrain_path)
        validate(model_path, xtest_path, ytest_path)

if __name__ == "__main__":
    fire.Fire(workflow)