import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from fastmcp import Client


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
        mcp_server_host = os.getenv(
            "MCP_SERVER_HOST", "http://titanic-mcp-server.gthomas59800-dev.svc.cluster.local:8000"
        )
        self.mcp_config = {
            "mcpServers": {
                "titanic": {
                    "url": f"{mcp_server_host}/mcp",
                    "transport": "streamable-http",
                }
            }
        }
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://models.github.ai/inference"),
            temperature=0.7,
        )

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Appelle un tool MCP et extrait le résultat texte."""
        async with Client(self.mcp_config) as mcp_client:
            await mcp_client.initialize()
            result = await mcp_client.call_tool(tool_name, arguments=arguments)

            if hasattr(result, "content") and result.content:
                content = result.content[0]
                if hasattr(content, "text"):
                    return content.text
                return str(content)
            return str(result)

    async def chat_async(self, message: str) -> str:
        """Chat async qui charge les tools MCP et les utilise via le LLM."""
        async with Client(self.mcp_config) as mcp_client:
            await mcp_client.initialize()
            tools_list = await mcp_client.list_tools()

            langchain_tools = [
                StructuredTool.from_function(
                    func=lambda: None,
                    name=tool.name,
                    description=tool.description,
                )
                for tool in tools_list
            ]

            llm_with_tools = self.llm.bind_tools(langchain_tools)

            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
            response = llm_with_tools.invoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                return await self._call_mcp_tool(tool_call["name"], tool_call["args"])

            return response.content

    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
