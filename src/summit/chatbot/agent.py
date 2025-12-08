import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from summit.chatbot.mcp_tools import TitanicInferenceTool


class ChatbotAgent:
    def __init__(self, api_url: str):
        self.titanic_tool = TitanicInferenceTool(api_url)
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
            temperature=0.7
        )
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _create_tools(self):
        @tool
        def predict_titanic_survival(pclass: int, sex: str, sibsp: int, parch: int) -> str:
            """Predict if a Titanic passenger would survive based on their characteristics.

            Args:
                pclass: Passenger class (1=Upper, 2=Middle, 3=Low)
                sex: Gender ('male' or 'female')
                sibsp: Number of siblings/spouses aboard
                parch: Number of parents/children aboard
            """
            result = self.titanic_tool.predict_survival(pclass, sex, sibsp, parch)

            if "error" in result:
                return f"Error: {result['error']}"

            survived = "survived" if result["survived"] else "did not survive"
            return f"The passenger {survived}. Prediction value: {result['prediction']}"

        return [predict_titanic_survival]

    def chat(self, message: str) -> str:
        try:
            messages = [
                SystemMessage(content="""You are a helpful assistant that predicts Titanic passenger survival.
When asked about predictions, use the predict_titanic_survival tool.

Passenger class: 1=Upper, 2=Middle, 3=Low
Gender: male or female

Be friendly and explain predictions clearly."""),
                HumanMessage(content=message)
            ]

            response = self.llm_with_tools.invoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "predict_titanic_survival":
                    result = self.titanic_tool.predict_survival(
                        tool_args["pclass"],
                        tool_args["sex"],
                        tool_args["sibsp"],
                        tool_args["parch"]
                    )

                    if "error" in result:
                        return f"Error making prediction: {result['error']}"

                    survived = "survived ✅" if result["survived"] else "did not survive ❌"
                    return f"Based on the characteristics provided, the passenger {survived}."

            return response.content

        except Exception as e:
            return f"Error: {str(e)}"

