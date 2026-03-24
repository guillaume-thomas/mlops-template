import os
import httpx
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from collections.abc import Callable, Awaitable
from fastmcp.server.dependencies import get_http_headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from opentelemetry import context as otel_context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import extract, inject, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from summit.mcp_server.auth import token_manager

API_URL = os.getenv("TITANIC_API_URL", "http://mlops-api-service.gthomas59800-dev.svc.cluster.local:8080")
JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.gthomas59800-dev.svc.cluster.local:4318/v1/traces")

resource = Resource(attributes={"service.name": "titanic-mcp-server"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(HTTPSpanExporter(endpoint=JAEGER_ENDPOINT))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
set_global_textmap(CompositePropagator([TraceContextTextMapPropagator()]))

tracer = trace.get_tracer(__name__)

mcp = FastMCP("titanic-mcp-server")


class OtelMiddleware(Middleware):
    """Extrait le traceparent W3C des headers HTTP entrants via le middleware FastMCP natif."""

    async def on_request(self, ctx: MiddlewareContext, call_next: Callable[..., Awaitable[object]]) -> object:  # type: ignore[override]
        headers = get_http_headers() or {}
        otel_ctx = extract(dict(headers))
        token = otel_context.attach(otel_ctx)
        try:
            return await call_next(ctx)
        finally:
            otel_context.detach(token)


mcp.add_middleware(OtelMiddleware())


@mcp.tool()
async def predict_survival(pclass: int, sex: str, sibsp: int, parch: int) -> str:
    """
    Prédit la survie d'un passager du Titanic.

    Args:
        pclass: Classe du billet (1, 2 ou 3)
        sex: Sexe ("male" ou "female")
        sibsp: Nombre de frères/sœurs/conjoints à bord
        parch: Nombre de parents/enfants à bord

    Returns:
        Prédiction de survie avec message et détails

    """
    with tracer.start_as_current_span("mcp.predict_survival") as span:
        span.set_attribute("passenger.pclass", pclass)
        span.set_attribute("passenger.sex", sex)
        span.set_attribute("passenger.sibsp", sibsp)
        span.set_attribute("passenger.parch", parch)

        try:
            payload = {"pclass": pclass, "sex": sex, "sibSp": sibsp, "parch": parch}

            headers: dict[str, str] = {"Content-Type": "application/json"}

            # Propagation W3C TraceContext vers l'API Titanic
            inject(headers)

            token = await token_manager.get_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{API_URL}/infer", json=payload, headers=headers, timeout=10.0)
                resp.raise_for_status()
                result = resp.json()

            prediction = result[0] if isinstance(result, list) else result
            survived = bool(prediction)
            span.set_attribute("prediction.result", int(prediction))

            if survived:
                return (
                    f"Good news! According to the prediction model, this passenger would have SURVIVED the Titanic "
                    f"disaster (prediction: {prediction})."
                )
            else:
                return (
                    f"Unfortunately, according to the prediction model, this passenger would NOT have survived the "
                    f"Titanic disaster (prediction: {prediction})."
                )
        except Exception as e:
            span.record_exception(e)
            return f"Sorry, I encountered an error while trying to predict: {e!s}"


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Health check endpoint pour Kubernetes."""
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport="streamable-http", host=host, port=port, path="/mcp")
