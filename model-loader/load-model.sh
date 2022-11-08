#!/bin/bash

MODELS_CONFIG_DIR=${1}

echo "Downloading the model from HuggingFace, this will take a while..."

MODEL_NUM_GPUS_NAME="${MODEL}-${NUM_GPUS}gpu"
MODEL_ARCHIVE="${MODELS_DIR}/${MODEL_NUM_GPUS_NAME}.tar.zst"

mkdir -p "${MODELS_DIR}"
cp -r "${MODELS_CONFIG_DIR}/${MODEL_NUM_GPUS_NAME}/." "${MODELS_DIR}"

curl -L "https://huggingface.co/moyix/${MODEL}-gptj/resolve/main/${MODEL}-${NUM_GPUS}gpu.tar.zst" -o "$MODEL_ARCHIVE"
zstd -dc "$MODEL_ARCHIVE" | tar -xf - -C "${MODELS_DIR}" --strip-components=1
rm -f "$MODEL_ARCHIVE"
