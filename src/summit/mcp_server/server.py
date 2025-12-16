import os
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route, Mount
from starlette.applications import Starlette
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from summit.mcp_server.titanic_tool import TitanicInferenceTool
import uvicorn

API_URL = os.getenv("TITANIC_API_URL", "http://mlops-api-service.gthomas59800-dev.svc.cluster.local:8080")

mcp = Server("titanic-mcp-server")
sse = SseServerTransport("/messages")
titanic_tool = TitanicInferenceTool(API_URL)


@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="predict_survival",
            description="Predict if a Titanic passenger would survive",
            inputSchema={
                "type": "object",
                "properties": {
                    "pclass": {"type": "integer", "description": "Passenger class (1, 2, or 3)"},
                    "sex": {"type": "string", "description": "Gender (male or female)"},
                    "sibsp": {"type": "integer", "description": "Number of siblings/spouses aboard"},
                    "parch": {"type": "integer", "description": "Number of parents/children aboard"},
                },
                "required": ["pclass", "sex", "sibsp", "parch"],
            },
        )
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "predict_survival":
        raise ValueError(f"Unknown tool: {name}")

    result = titanic_tool.predict_survival(**arguments)

    if "error" in result:
        text = f"Error: {result['error']}"
    else:
        status = "survived" if result["survived"] else "did not survive"
        text = f"Based on the passenger characteristics, they {status}. (Prediction: {result['prediction']})"

    return [types.TextContent(type="text", text=text)]


async def sse_handler(request: Request) -> Response:
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())
    return Response()


async def health_handler(request: Request) -> Response:
    return Response(content='{"status":"healthy"}', media_type="application/json")


app = Starlette(
    routes=[
        Route("/sse", endpoint=sse_handler, methods=["GET"]),
        Mount("/messages", app=sse.handle_post_message),
        Route("/health", endpoint=health_handler, methods=["GET"]),
    ]
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
