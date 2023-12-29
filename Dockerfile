FROM python:3.10.13-slim AS base-image

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  POETRY_VERSION=1.7.1

WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install "poetry==$POETRY_VERSION"

# Install all dependencies into /opt/venv
# so that we can copy these resources between
# build stages
RUN poetry config virtualenvs.path /opt/venv

##
## Intermediate image contains build-essential for installing
## google-cloud-profiler's dependencies
##
FROM base-image AS install-image

ENV DEBIAN_FRONTEND=noninteractive

COPY scripts /app/scripts

# Install tree-sitter build dependencies
RUN /app/scripts/install-tree-sitter-deps

RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main

# Workaround to resolve https://github.com/orgs/community/discussions/55820
RUN git config --global http.version HTTP/1.1

# Build tree-sitter library for the grammars supported
RUN poetry run python /app/scripts/build-tree-sitter-lib.py

##
## Final image copies dependencies from install-image
##
FROM base-image as final

COPY --from=install-image /opt/venv /opt/venv
COPY --from=install-image /app/scripts/lib/*.so /usr/lib

COPY ai_gateway/ ai_gateway/

# Environment variable TRANSFORMERS_CACHE controls where files are downloaded
COPY --from=install-image /app/scripts/bootstrap.py .
RUN poetry run python bootstrap.py

CMD ["poetry", "run", "ai_gateway"]
