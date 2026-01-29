import os
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
from langchain_mcp_adapters.client import MultiServerMCPClient


SYSTEM_PROMPT = """You are a helpful assistant that predicts Titanic passenger survival.

To make a prediction, use the predict_survival tool with ALL required parameters:
- pclass (integer): Passenger class - 1 (First), 2 (Second), or 3 (Third)
- sex (string): "male" or "female"
- sibsp (integer): Number of siblings/spouses aboard (0-8)
- parch (integer): Number of parents/children aboard (0-9)

If the user doesn't specify all parameters, ask politely for missing information.
NEVER guess values - always ask the user.

Examples:
- "A man" → Ask: "What class? Any family aboard?"
- "A man in third class alone" → Use: pclass=3, sex="male", sibsp=0, parch=0

Be friendly and explain predictions clearly."""


class ChatbotAgent:
    def __init__(self) -> None:
        mcp_server_host: str = os.getenv(
            "MCP_SERVER_HOST", "http://titanic-mcp-server.gthomas59800-dev.svc.cluster.local:8000"
        )

        self.mcp_connections = {"titanic": {"url": f"{mcp_server_host}/mcp", "transport": "streamable_http"}}

        api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=SecretStr(api_key),
            base_url=os.getenv("OPENAI_BASE_URL", "https://models.github.ai/inference"),
            temperature=0.7,
        )

    async def chat_async(self, message: str) -> str:
        """Chat async utilisant l'adaptateur MCP officiel."""
        mcp_client = MultiServerMCPClient(self.mcp_connections)  # type: ignore

        tools = await mcp_client.get_tools()
        llm_with_tools = self.llm.bind_tools(tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
        response = await llm_with_tools.ainvoke(messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            for tool in tools:
                if tool.name == tool_name:
                    result = await tool.ainvoke(tool_args)
                    if hasattr(result, "content") and result.content:
                        content = result.content[0]
                        if hasattr(content, "text"):
                            return content.text
                        return str(content)
                    return str(result)

        return str(response.content)

    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
