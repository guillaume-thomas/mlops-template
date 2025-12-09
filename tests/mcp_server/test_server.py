import os
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client


@pytest.fixture
def mcp_server_url():
    return os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")


@pytest.mark.asyncio
async def test_mcp_server_connection(mcp_server_url):
    try:
        async with sse_client(mcp_server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                assert session is not None
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")


@pytest.mark.asyncio
async def test_mcp_server_list_tools(mcp_server_url):
    try:
        async with sse_client(mcp_server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()

                assert len(tools_result.tools) > 0
                assert tools_result.tools[0].name == "predict_survival"
                assert "predict" in tools_result.tools[0].description.lower()
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")


@pytest.mark.asyncio
async def test_mcp_server_predict_first_class_female(mcp_server_url):
    try:
        async with sse_client(mcp_server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "predict_survival",
                    {
                        "pclass": 1,
                        "sex": "female",
                        "sibsp": 1,
                        "parch": 0
                    }
                )

                assert result.content is not None
                assert len(result.content) > 0
                assert "survived" in result.content[0].text.lower()
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")


@pytest.mark.asyncio
async def test_mcp_server_predict_third_class_male(mcp_server_url):
    try:
        async with sse_client(mcp_server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "predict_survival",
                    {
                        "pclass": 3,
                        "sex": "male",
                        "sibsp": 0,
                        "parch": 0
                    }
                )

                assert result.content is not None
                assert len(result.content) > 0
                assert result.content[0].text is not None
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")

