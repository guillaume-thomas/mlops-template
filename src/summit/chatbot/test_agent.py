import os
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TITANIC_API_URL"] = "http://localhost:8080"

from summit.chatbot.agent import ChatbotAgent

agent = ChatbotAgent("http://localhost:8080")
print("✅ Agent créé avec succès")
print(f"✅ LLM configuré: {agent.llm.model_name}")
print(f"✅ Outils disponibles: {len(agent.tools)}")
print(f"✅ Outil: {agent.tools[0].name}")

