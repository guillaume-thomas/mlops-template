FROM python:3.13-slim

ARG USER_UID=1001

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY --chown=${USER_UID}:${USER_UID} pyproject.toml uv.lock .python-version ./
COPY --chown=${USER_UID}:${USER_UID} ./src/summit/api ./src/summit/api

RUN uv sync -n --no-group training --group api

EXPOSE 8080

USER ${USER_UID}

ENTRYPOINT ["uv", "-n", "run", "--no-sync", "api"]