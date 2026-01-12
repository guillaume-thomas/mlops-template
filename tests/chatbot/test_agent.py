import os
import pytest
import inspect

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["MCP_SERVER_HOST"] = "http://localhost:8000"

from summit.chatbot.agent import ChatbotAgent


@pytest.fixture
def agent():
    return ChatbotAgent()


def test_agent_mcp_config_structure(agent):
    """Test que la configuration MCP a la bonne structure."""
    assert agent.mcp_config is not None
    assert "mcpServers" in agent.mcp_config
    assert "titanic" in agent.mcp_config["mcpServers"]

    titanic_config = agent.mcp_config["mcpServers"]["titanic"]
    assert titanic_config["url"] == "http://localhost:8000/mcp"
    assert titanic_config["transport"] == "streamable-http"


def test_agent_llm_uses_correct_model(agent):
    """Test que le LLM utilise le bon modèle et la bonne configuration."""
    assert agent.llm is not None

    if hasattr(agent.llm, "model_name"):
        assert "gpt-4o-mini" in agent.llm.model_name

    if hasattr(agent.llm, "temperature"):
        assert agent.llm.temperature == 0.7


def test_agent_chat_method_signature(agent):
    """Test que la méthode chat a la bonne signature."""
    assert hasattr(agent, "chat")
    assert callable(agent.chat)

    sig = inspect.signature(agent.chat)
    assert "message" in sig.parameters
    assert sig.return_annotation is str


def test_agent_uses_environment_variables():
    """Test que l'agent utilise correctement les variables d'environnement."""
    os.environ["MCP_SERVER_HOST"] = "http://custom-host:9000"

    agent = ChatbotAgent()

    assert "http://custom-host:9000/mcp" in agent.mcp_config["mcpServers"]["titanic"]["url"]

    os.environ["MCP_SERVER_HOST"] = "http://localhost:8000"
