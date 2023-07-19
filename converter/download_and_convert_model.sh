#!/bin/bash

MODEL_ARCH=${1}
SAVE_DIR=${2}

python3 codegen_gptj_convert.py \
    --code_model Salesforce/"${MODEL_ARCH}" \
    tmp-gptj

python3 huggingface_gptj_convert.py \
    -in_file tmp-gptj \
    -saved_dir ${SAVE_DIR}/1 \
    -infer_gpu_num 1

python3 huggingface_gptj_convert.py \
    -in_file tmp-gptj \
    -saved_dir ${SAVE_DIR}/2 \
    -infer_gpu_num 2

rm -rf tmp-gptj
