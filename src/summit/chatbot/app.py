import os
import streamlit as st
from summit.chatbot.agent import ChatbotAgent


def main() -> None:
    st.set_page_config(page_title="Titanic Survival Chatbot", page_icon="🚢", layout="centered")

    st.title("🚢 Titanic Survival Prediction Chatbot")
    st.markdown("Ask me about Titanic passenger survival predictions!")

    if "agent" not in st.session_state:
        st.session_state.agent = ChatbotAgent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Titanic survival predictions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **LangChain** for agent orchestration
        - **GitHub Models** (gpt-4o-mini) - FREE with GitHub account
        - **MCP Tools** to call Titanic inference API

        **Example questions:**
        - "Would a first-class female passenger with 1 sibling, no parent, no child, survive?"
        - "Predict survival for a third-class male with no family"
        - "What are the survival chances for a middle-class, single with no sibling woman?"
        """)

        st.markdown("---")
        st.markdown("**Configuration:**")
        st.code(f"LLM: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}", language=None)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
