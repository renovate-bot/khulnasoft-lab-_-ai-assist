ARG SRC_IMAGE="registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:latest"

FROM ${SRC_IMAGE}
ARG TAG

WORKDIR /app

RUN pip install langchain=="0.3.3" langchain_text_splitters=="0.3.0"

RUN mkdir "tmp"

# Specific configurations for self-hosted models
ENV AIGW_CUSTOM_MODELS__ENABLED=true
ENV AIGW_FASTAPI__API_PORT=5052
ENV AIGW_FASTAPI__OPENAPI_URL="/openapi.json"
ENV AIGW_FASTAPI__DOCS_URL="/docs"

COPY scripts/index_docs_as_sqlite.py index_docs_as_sqlite.py
COPY scripts/troubleshoot_selfhosted_installation.py troubleshoot_selfhosted_installation.py
RUN poetry run python index_docs_as_sqlite.py -o "./tmp/docs.db" -v ${TAG##self-hosted-}

CMD ["./run.sh"]
