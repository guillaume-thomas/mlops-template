import os
import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from summit.mcp_server.auth import token_manager

API_URL = os.getenv("TITANIC_API_URL", "http://mlops-api-service.gthomas59800-dev.svc.cluster.local:8080")

mcp = FastMCP("titanic-mcp-server")


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
    try:
        payload = {"pclass": pclass, "sex": sex, "sibSp": sibsp, "parch": parch}

        headers = {"Content-Type": "application/json"}
        token = await token_manager.get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_URL}/infer", json=payload, headers=headers, timeout=10.0)
            resp.raise_for_status()
            result = resp.json()

        prediction = result[0] if isinstance(result, list) else result
        survived = bool(prediction)

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
        return f"Sorry, I encountered an error while trying to predict: {e!s}"


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Health check endpoint pour Kubernetes."""
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport="streamable-http", host=host, port=port, path="/mcp")
