import fire
import mlflow
from mlflow.tracking import MlflowClient

def workflow(input_data_path: str):
    print(f"workflow input path : {input_data_path}")
    client = MlflowClient()
    load_data_run = mlflow.run(
        ".", "load_data", parameters={"path": input_data_path}, env_manager="local"
    )

    # Wait for completion and get output
    if load_data_run.wait():
        load_data_run_instance = client.get_run(load_data_run.run_id)
        out_path = load_data_run_instance.info.artifact_uri + "/path_output/output.csv"
        split_train_test_run = mlflow.run(
            ".", "split_train_test", parameters={"output_path": out_path}, env_manager="local"
        )

        if split_train_test_run.wait():
            split_train_test_run_instance = client.get_run(split_train_test_run.run_id)
            artifact_uri = split_train_test_run_instance.info.artifact_uri
            train_model_run = mlflow.run(
                ".", "train",
                parameters={"xtrain": artifact_uri + "/xtrain/xtrain.csv", "ytrain": artifact_uri + "/ytrain/ytrain.csv"},
                env_manager="local"
            )

            if train_model_run.wait():
                train_model_run_instance = client.get_run(train_model_run.run_id)
                train_model_artifact_uri = train_model_run_instance.info.artifact_uri
                print(train_model_artifact_uri)
                """
                mlflow.run(
                    ".", "validation",
                    parameters={"model_path": train_model_artifact_uri + "/model/model.joblib",
                                "xtest": artifact_uri + "/xtest/xtest.csv",
                                "ytest": artifact_uri + "/ytest/ytest.csv"
                                },
                    env_manager="local"
                )
                """



if __name__ == "__main__":
    fire.Fire(workflow)