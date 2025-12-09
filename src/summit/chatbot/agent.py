import os
import asyncio
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import StructuredTool


class ChatbotAgent:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://titanic-mcp-server.gthomas59800-dev.svc.cluster.local:8000/sse")
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
            temperature=0.7
        )
        self.mcp_session: Optional[ClientSession] = None
        self.mcp_tools = []
        self._loop = None

    async def _init_mcp_client(self):
        try:
            self._sse_context = sse_client(self.mcp_server_url)
            self.read_stream, self.write_stream = await asyncio.wait_for(
                self._sse_context.__aenter__(),
                timeout=10.0
            )

            self.mcp_session = ClientSession(self.read_stream, self.write_stream)
            await self.mcp_session.__aenter__()

            await asyncio.wait_for(self.mcp_session.initialize(), timeout=10.0)

            result = await asyncio.wait_for(self.mcp_session.list_tools(), timeout=10.0)
            self.mcp_tools = result.tools

            self.langchain_tools = self._convert_mcp_tools_to_langchain()
            self.llm_with_tools = self.llm.bind_tools(self.langchain_tools)

        except asyncio.TimeoutError as e:
            raise Exception(f"MCP server connection timeout. Is the server running at {self.mcp_server_url}?")
        except Exception as e:
            raise

    def _convert_mcp_tools_to_langchain(self):
        langchain_tools = []
        for mcp_tool in self.mcp_tools:
            def create_tool_func(tool_name):
                async def tool_func(**kwargs) -> str:
                    try:
                        result = await self.mcp_session.call_tool(tool_name, kwargs)
                        return result.content[0].text if result.content else "No result"
                    except Exception as e:
                        return f"Error calling tool: {str(e)}"
                return tool_func

            structured_tool = StructuredTool.from_function(
                func=create_tool_func(mcp_tool.name),
                name=mcp_tool.name,
                description=mcp_tool.description,
                coroutine=create_tool_func(mcp_tool.name)
            )
            langchain_tools.append(structured_tool)

        return langchain_tools

    async def _chat_async(self, message: str) -> str:
        if not self.mcp_session:
            await self._init_mcp_client()

        try:
            messages = [
                SystemMessage(content="""You are a helpful assistant that predicts Titanic passenger survival.

To make a prediction, you MUST use the predict_survival tool with ALL required parameters:

REQUIRED parameters:
- pclass (integer): Passenger class - 1 (First/Upper), 2 (Second/Middle), or 3 (Third/Lower)
- sex (string): Gender - "male" or "female"
- sibsp (integer): Number of siblings/spouses aboard (0-8)
- parch (integer): Number of parents/children aboard (0-9)

IMPORTANT: 
- If the user doesn't specify all parameters, ask them politely for the missing information.
- NEVER guess or assume values - always ask the user.
- sibsp = 0 means traveling alone (no siblings/spouses)
- parch = 0 means traveling alone (no parents/children)

Examples:
- "A man" → Ask: "What class was he in? Did he have family aboard?"
- "A woman in first class" → Ask: "Did she have any siblings, spouses, parents, or children with her?"
- "A man in third class alone" → Use: pclass=3, sex="male", sibsp=0, parch=0

Be friendly and explain predictions clearly with probabilities."""),
                HumanMessage(content=message)
            ]

            response = self.llm_with_tools.invoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                result = await self.mcp_session.call_tool(tool_name, tool_args)
                result_text = result.content[0].text if result.content else "No result"

                return result_text

            return response.content

        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self, message: str) -> str:
        if not self._loop:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        try:
            return self._loop.run_until_complete(self._chat_async(message))
        except Exception as e:
            return f"Error: {str(e)}"

    async def close(self):
        try:
            if self.mcp_session:
                await self.mcp_session.__aexit__(None, None, None)
            if hasattr(self, '_sse_context'):
                await self._sse_context.__aexit__(None, None, None)
        except Exception as e:
            pass

