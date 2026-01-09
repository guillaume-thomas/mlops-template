import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession, Tool
from mcp.client.sse import sse_client


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
        self.mcp_url = os.getenv(
            "MCP_SERVER_URL", "http://titanic-mcp-server.gthomas59800-dev.svc.cluster.local:8000/sse"
        )
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://models.github.ai/inference"),
            temperature=0.7,
        )
        self.mcp_session = None
        self._sse_context = None
        self._loop = None

    async def _init_mcp(self) -> None:
        self._sse_context = sse_client(self.mcp_url)
        read_stream, write_stream = await asyncio.wait_for(self._sse_context.__aenter__(), timeout=10.0)

        self.mcp_session = ClientSession(read_stream, write_stream)
        await self.mcp_session.__aenter__()
        await asyncio.wait_for(self.mcp_session.initialize(), timeout=10.0)

        tools_result = await asyncio.wait_for(self.mcp_session.list_tools(), timeout=10.0)
        langchain_tools = [self._create_langchain_tool(t) for t in tools_result.tools]
        self.llm = self.llm.bind_tools(langchain_tools)

    def _create_langchain_tool(self, mcp_tool: Tool) -> StructuredTool:
        async def call_mcp(**kwargs: dict[str, Any]) -> str:
            result = await self.mcp_session.call_tool(mcp_tool.name, kwargs)
            return result.content[0].text if result.content else "No result"

        return StructuredTool.from_function(
            func=call_mcp, name=mcp_tool.name, description=mcp_tool.description, coroutine=call_mcp
        )

    async def _chat_async(self, message: str) -> str:
        if not self.mcp_session:
            await self._init_mcp()

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]

        response = self.llm.invoke(messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            result = await self.mcp_session.call_tool(tool_call["name"], tool_call["args"])
            return result.content[0].text if result.content else "No result"

        return response.content

    def chat(self, message: str) -> str:
        if not self._loop:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop.run_until_complete(self._chat_async(message))

    async def close(self) -> None:
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if self._sse_context:
            await self._sse_context.__aexit__(None, None, None)
