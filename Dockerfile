FROM python:3.13.3-slim AS base-image

# HF_HOME and POETRY_* are needed to allow the docker image to be run from a non-root users
# without having issues with permissions
ENV PYTHONUNBUFFERED=1 \
  HF_HOME=/home/aigateway/.hf \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  POETRY_VERSION=1.8.5 \
  POETRY_VIRTUALENVS_PATH=/home/aigateway/app/venv \
  POETRY_CONFIG_DIR=/home/aigateway/app/.config/pypoetry \
  POETRY_DATA_DIR=/home/aigateway/app/.local/share/pypoetry \
  POETRY_CACHE_DIR=/home/aigateway/app/.cache/pypoetry \
  CLOUD_CONNECTOR_SERVICE_NAME=${CLOUD_CONNECTOR_SERVICE_NAME}

WORKDIR /home/aigateway/app

COPY poetry.lock pyproject.toml ./
RUN pip install "poetry==$POETRY_VERSION"
RUN mkdir -p -m 777 $POETRY_CONFIG_DIR $POETRY_DATA_DIR $POETRY_CACHE_DIR

##
## Intermediate image contains build-essential for installing
## google-cloud-profiler's dependencies
##
FROM base-image AS install-image

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY scripts /home/aigateway/app/scripts
COPY vendor /home/aigateway/app/vendor

RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main

##
## Final image copies dependencies from install-image
##
FROM base-image AS final

WORKDIR /home/aigateway/app

RUN useradd aigateway
RUN chown -R aigateway:aigateway /home/aigateway/
USER aigateway

COPY --chown=aigateway:aigateway --from=install-image /home/aigateway/app/venv/ /home/aigateway/app/venv/

COPY --chown=aigateway:aigateway ai_gateway/ ai_gateway/
COPY --chown=aigateway:aigateway vendor/ /home/aigateway/app/vendor/

# Environment variable TRANSFORMERS_CACHE controls where files are downloaded
COPY --chown=aigateway:aigateway --from=install-image /home/aigateway/app/scripts/bootstrap.py .
COPY --chown=aigateway:aigateway --from=install-image /home/aigateway/app/scripts/run.sh .

RUN poetry run python bootstrap.py

EXPOSE 5052

CMD ["./run.sh"]
