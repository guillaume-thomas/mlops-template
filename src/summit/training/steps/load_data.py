import logging
import os
from pathlib import Path
import tempfile

import boto3
import fire
import mlflow
import pandas as pd
from ydata_profiling import ProfileReport

ARTIFACT_PATH = "path_output"
PROFILING_PATH = "profiling_reports"


def load_data(path: str) -> str:
    logging.warning(f"load_data on path : {path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir, "data.csv")
        logging.warning(f"to path : {local_path}")

        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        s3_client.download_file("summit", path, local_path)
        df = pd.read_csv(local_path)

        profile = ProfileReport(df, title=f"Profiling Report - {local_path.stem}")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            profile.to_file(tmp_file.name)
            mlflow.log_artifact(tmp_file.name, PROFILING_PATH)
            os.unlink(tmp_file.name)

        mlflow.log_artifact(str(local_path), ARTIFACT_PATH)

    return f"{ARTIFACT_PATH}/{local_path.name}"


if __name__ == "__main__":
    fire.Fire(load_data)
