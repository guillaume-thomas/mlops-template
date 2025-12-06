import os
import pickle
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


# Set the endpoint and header
#MLFLOW_TRACKING_URI = "mlflow.gthomas59800-dev.svc.cluster.local"
MLFLOW_TRACKING_URI = "http://mlflow-gthomas59800-dev.apps.rm3.7wse.p1.openshiftapps.com"
MLFLOW_EXPERIMENT_ID = "1" # Trouver un moyen de le passer dynamiquement

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = f"{MLFLOW_TRACKING_URI}/v1/traces"
os.environ[
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
] = f"x-mlflow-experiment-id={MLFLOW_EXPERIMENT_ID}"


app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

model = pickle.load(open("./src/summit/api/resources/model.pkl", "rb"))


class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3

class Sex(Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self):
        return {
            "Pclass": self.pclass.value,
            "Sex": self.sex.value,
            "SibSp": self.sibSp,
            "Parch": self.parch
        }

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/infer")
def infer(passenger: Passenger) -> list:
    df_passenger = pd.DataFrame([passenger.to_dict()])
    df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
    df_to_predict = pd.get_dummies(df_passenger)
    res = model.predict(df_to_predict)
    return res.tolist()