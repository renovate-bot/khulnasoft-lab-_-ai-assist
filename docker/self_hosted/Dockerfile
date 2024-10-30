ARG SRC_IMAGE="registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:latest"

FROM ${SRC_IMAGE}
ARG TAG
RUN mkdir "tmp"

# Specific configurations for self-hosted models
ENV AIGW_CUSTOM_MODELS__ENABLED=true
ENV AIGW_FASTAPI__API_PORT=5052
ENV AIGW_FASTAPI__OPENAPI_URL="/openapi.json"
ENV AIGW_FASTAPI__DOCS_URL="/docs"

RUN poetry run index_docs -o "./tmp/docs.db" -v ${TAG##self-hosted-}

CMD ["./run.sh"]
