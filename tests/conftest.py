import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["JAEGER_ENDPOINT"] = "http://localhost:4318/v1/traces"

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import opentelemetry
    import opentelemetry.sdk.trace.export
    import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401

    HAS_OPENTELEMETRY = True
except (ImportError, AttributeError):
    HAS_OPENTELEMETRY = False


@pytest.fixture(scope="session", autouse=True)
def setup_mlflow_tracking() -> None:
    """Configure MLflow pour utiliser un répertoire temporaire pendant les tests."""
    if not HAS_MLFLOW:
        yield
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.set_tracking_uri(f"file://{tmp_dir}/mlruns")
        yield
        mlflow.set_tracking_uri(None)


@pytest.fixture(autouse=True)
def cleanup_mlflow_artifacts() -> None:
    """Nettoie les artifacts MLflow après chaque test."""
    yield

    if not HAS_MLFLOW:
        return

    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists():
        shutil.rmtree(mlruns_dir)

    mlflow_db = Path("mlflow.db")
    if mlflow_db.exists():
        mlflow_db.unlink()


@pytest.fixture(scope="session", autouse=True)
def mock_opentelemetry() -> None:
    """Mock OpenTelemetry pour éviter les connexions réseau dans les tests."""
    if not HAS_OPENTELEMETRY:
        yield
        return

    mock_processor = MagicMock()
    mock_processor.force_flush = Mock(return_value=True)
    mock_processor.shutdown = Mock(return_value=True)

    mock_exporter = MagicMock()
    mock_exporter.export = Mock(return_value=MagicMock())
    mock_exporter.shutdown = Mock(return_value=True)

    mock_tracer_provider = MagicMock()
    mock_tracer = MagicMock()
    mock_span = MagicMock()

    mock_span.__enter__ = Mock(return_value=mock_span)
    mock_span.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span = Mock(return_value=mock_span)
    mock_tracer_provider.get_tracer = Mock(return_value=mock_tracer)

    with (
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", return_value=mock_processor),
        patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", return_value=mock_exporter),
        patch("opentelemetry.sdk.trace.TracerProvider", return_value=mock_tracer_provider),
        patch("opentelemetry.trace.set_tracer_provider"),
        patch("opentelemetry.trace.get_tracer_provider", return_value=mock_tracer_provider),
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
        patch("requests.post", return_value=MagicMock(status_code=200)),
        patch("httpx.post", return_value=MagicMock(status_code=200)),
    ):
        yield
