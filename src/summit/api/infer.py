import os
import pickle
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from fastapi import FastAPI, Depends
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource

from summit.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.gthomas59800-dev.svc.cluster.local:4318/v1/traces")

resource = Resource(attributes={"service.name": "titanic-inference-api"})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(HTTPSpanExporter(endpoint=JAEGER_ENDPOINT))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

with open("./src/summit/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)


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

    def to_dict(self) -> dict:
        return {"Pclass": self.pclass.value, "Sex": self.sex.value, "SibSp": self.sibSp, "Parch": self.parch}


@app.get("/health")
def health() -> dict:
    return {"status": "OK"}


@app.post("/infer")
def infer(passenger: Passenger, token: str = Depends(verify_token("api:read"))) -> list:
    with tracer.start_as_current_span("model_inference") as span:
        span.set_attribute("passenger.pclass", passenger.pclass.value)
        span.set_attribute("passenger.sex", passenger.sex.value)
        span.set_attribute("passenger.sibsp", passenger.sibSp)
        span.set_attribute("passenger.parch", passenger.parch)

        df_passenger = pd.DataFrame([passenger.to_dict()])
        df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
        df_to_predict = pd.get_dummies(df_passenger)

        res = model.predict(df_to_predict)

        span.set_attribute("prediction.result", int(res[0]))
        span.add_event("prediction_completed", {"result": int(res[0])})

        return res.tolist()
