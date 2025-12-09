import os
import logging
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route, Mount
from starlette.applications import Starlette
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from summit.mcp_server.titanic_tool import TitanicInferenceTool
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("TITANIC_API_URL", "http://mlops-api-service.gthomas59800-dev.svc.cluster.local:8080")
titanic_tool = TitanicInferenceTool(API_URL)

mcp = Server("titanic-mcp-server")
sse = SseServerTransport("/messages")

@mcp.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    logger.info("[MCP] list_tools called")
    tools = [
        types.Tool(
            name="predict_survival",
            description="Predict if a Titanic passenger would survive based on their characteristics",
            inputSchema={
                "type": "object",
                "properties": {
                    "pclass": {"type": "integer", "description": "Passenger class (1=Upper, 2=Middle, 3=Low)"},
                    "sex": {"type": "string", "description": "Gender (male or female)"},
                    "sibsp": {"type": "integer", "description": "Number of siblings/spouses aboard"},
                    "parch": {"type": "integer", "description": "Number of parents/children aboard"}
                },
                "required": ["pclass", "sex", "sibsp", "parch"]
            }
        )
    ]
    logger.info(f"[MCP] Returning {len(tools)} tools")
    return tools

@mcp.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info(f"[MCP] call_tool: {name} with args {arguments}")
    if name != "predict_survival":
        raise ValueError(f"Unknown tool: {name}")

    result = titanic_tool.predict_survival(**arguments)
    logger.info(f"[MCP] Tool result: {result}")

    if "error" in result:
        text = f"Error making prediction: {result['error']}"
    else:
        survived_text = "survived" if result["survived"] else "did not survive"
        text = f"Based on the passenger characteristics, they {survived_text}. (Prediction: {result['prediction']})"

    return [types.TextContent(type="text", text=text)]

async def handle_sse(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[SSE] Connection from {client_ip}")

    auth_header = request.headers.get("Authorization")
    if auth_header:
        logger.info(f"[SSE] Auth header present for {client_ip}")

    try:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            logger.info(f"[SSE] SSE connection established for {client_ip}")
            logger.info(f"[SSE] Starting MCP server for {client_ip}")

            await mcp.run(
                streams[0], streams[1], mcp.create_initialization_options()
            )

            logger.info(f"[SSE] MCP server stopped for {client_ip}")
    except Exception as e:
        logger.error(f"[SSE] Error for {client_ip}: {e}", exc_info=True)
        raise

    return Response()



async def health(request: Request):
    return Response(content='{"status":"healthy"}', media_type="application/json")


routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/messages", app=sse.handle_post_message),
    Route("/health", endpoint=health, methods=["GET"]),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
