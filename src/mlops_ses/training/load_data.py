import fire
import mlflow


def load_data(path: str):
    print(f"load_data on path : {path}")
    mlflow.log_artifact(path, "path_output")


if __name__ == "__main__":
    fire.Fire(load_data)
