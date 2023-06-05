FROM python:3.9.16-slim AS base-image

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  POETRY_VERSION=1.3

WORKDIR /app
RUN pip install "poetry==$POETRY_VERSION"
RUN pip install "lockfile"

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

RUN apt-get update

RUN apt-get install -y build-essential

COPY poetry.lock pyproject.toml ./

RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main 

## 
## Final image copies dependencies from install-image
## 
FROM base-image as final

COPY --from=install-image /opt/venv /opt/venv

COPY poetry.lock pyproject.toml ./
COPY codesuggestions/ codesuggestions/

CMD ["poetry", "run", "codesuggestions"]
