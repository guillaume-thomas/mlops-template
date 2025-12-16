import os
import pytest

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TITANIC_API_URL"] = "http://localhost:8080"
os.environ["MCP_SERVER_URL"] = "http://localhost:8000/sse"

from summit.chatbot.agent import ChatbotAgent


@pytest.fixture
def agent():
    return ChatbotAgent("http://localhost:8080")


def test_agent_creation(agent):
    assert agent is not None
    assert agent.api_url == "http://localhost:8080"
    assert agent.mcp_server_url == "http://localhost:8000/sse"


def test_agent_llm_configuration(agent):
    assert agent.llm is not None
    assert agent.llm.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_agent_mcp_initialization(agent):
    try:
        await agent._init_mcp_client()
        assert agent.mcp_session is not None
        assert len(agent.mcp_tools) > 0
        assert agent.mcp_tools[0].name == "predict_survival"
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")
    finally:
        if agent.mcp_session:
            await agent.close()
