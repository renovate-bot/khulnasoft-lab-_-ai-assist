FROM alpine:3.17.0

ARG MODEL_ID="codegen"
ARG MODEL_NAME="Salesforce/codegen-350M-multi"
ARG MODEL_VERSION="1"
ARG MODEL_HF_CACHE="/hf/${MODEL_NAME}"
ARG MODEL_OFFLOAD_FOLDER="/offload/${MODEL_NAME}"

ENV MODEL_HF_REPO="https://huggingface.co/${MODEL_NAME}"
ENV MODELS_REPO="/models"

RUN apk add --no-cache git git-lfs

RUN mkdir -p /model/${MODEL_ID}/${MODEL_VERSION} && mkdir -p ${MODELS_REPO} \
RUN mkdir -p ${MODELS_HF_CACHE} && mkdir -p ${MODEL_OFFLOAD_FOLDER}
WORKDIR /model/${MODEL_ID}

COPY ${MODEL_ID}/config.pbtxt.tmpl .
COPY ${MODEL_ID}/model.py ${MODEL_VERSION}/.

RUN sed -e s@{model_name}@"${MODEL_NAME}"@g \
		-e s@{model_path}@"${MODEL_HF_CACHE}"@g \
		-e s@{offload_folder}@"${MODEL_OFFLOAD_FOLDER}"@g \
		config.pbtxt.tmpl > config.pbtxt

ENTRYPOINT ["sh", "-c"]
