FROM python:3.9.16-slim

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  POETRY_VERSION=1.3 \
  POETRY_VIRTUALENVS_CREATE=false

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY poetry.lock pyproject.toml ./

RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main

COPY codesuggestions/ codesuggestions/

CMD ["poetry", "run", "codesuggestions"]
