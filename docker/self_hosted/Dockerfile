ARG SRC_IMAGE="registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:latest"

FROM ${SRC_IMAGE}
ARG TAG

WORKDIR /app

# Specific configurations for self-hosted models

RUN mkdir "tmp"
RUN pip install langchain langchain_text_splitters
COPY scripts/index_docs_as_sqlite.py index_docs_as_sqlite.py
RUN poetry run python index_docs_as_sqlite.py -o "./tmp/docs.db" -v ${TAG##self-hosted-}

ENV AIGW_CUSTOM_MODELS__ENABLED=true
ENV AIGW_FASTAPI__API_PORT=5052
ENV AIGW_FASTAPI__OPENAPI_URL="/openapi.json"
ENV AIGW_FASTAPI__DOCS_URL="/docs"

CMD ["./run.sh"]
