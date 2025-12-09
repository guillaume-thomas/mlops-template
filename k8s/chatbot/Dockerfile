FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock .python-version ./
COPY ./src/summit/chatbot ./src/summit/chatbot

ENV HOME=/tmp

RUN uv sync -n --group chatbot

EXPOSE 8501

ENTRYPOINT ["uv", "-n", "run", "--no-sync", "--group", "chatbot", "streamlit", "run", "src/summit/chatbot/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

