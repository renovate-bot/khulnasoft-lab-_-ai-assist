import os
import logging
from logging.config import dictConfig

import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG_LOGGER = {
    "version": 1,
    "formatters": {
        "default": {
            "format": '%(levelname)s %(asctime)s - "%(message)s"',
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "model.codegen": {
            "handlers": ["default"],
        },
    },
}

dictConfig(CONFIG_LOGGER)
log = logging.getLogger("model.codegen")
log.setLevel(logging.DEBUG if os.environ.get("MODEL_DEBUG") else logging.INFO)


def pb2numpy(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return tensor.as_numpy()


def numpy2pb(name, data):
    return pb_utils.Tensor(name, data)


class TritonPythonModel:
    MAX_PROMPT_LEN = 256

    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None

    def initialize(self, _: dict):
        model_path = os.environ.get("MODEL_PATH")
        if model_path is None:
            raise ValueError("MODEL_PATH env variable not set in the model config file")

        offload_folder = os.environ.get("OFFLOAD_FOLDER")
        if offload_folder is None:
            raise ValueError("OFFLOAD_FOLDER env variable not set in the model config file")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.padding_side = "left"

        # initialize model
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                offload_folder=offload_folder,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            prompts_np = pb2numpy(request, "prompts")
            prompt = prompts_np[0, 0].decode("utf-8")
            prompt_encoded = (
                self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=self.MAX_PROMPT_LEN,
                    truncation=True,
                    padding="max_length"
                )
                .to(self.device)
            )

            suggestion = self.model.generate(
                inputs=prompt_encoded["input_ids"],
                attention_mask=prompt_encoded["attention_mask"],
                max_new_tokens=20,
                top_k=3,
                penalty_alpha=0.4,
                pad_token_id=50256
            )

            # Take the code completion only instead of the prompt + completion
            completion = suggestion[:, self.MAX_PROMPT_LEN:]
            completion_decoded = self.tokenizer.batch_decode(
                completion,
                skip_special_tokens=True
            )

            log.debug(f"Prompt received: {prompt}")
            log.debug(f"Completion generated: {completion_decoded}")

            completions_np = np.array(completion_decoded, dtype="object")
            completions_pb = numpy2pb("completions", completions_np)

            response = pb_utils.InferenceResponse([completions_pb])
            responses.append(response)

        return responses
