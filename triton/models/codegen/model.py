import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM


def pb2numpy(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return tensor.as_numpy()


def numpy2pb(name, data):
    return pb_utils.Tensor(name, data)


class TritonPythonModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize(self, args: dict):
        model_config = json.loads(args["model_config"])
        model_path = model_config["parameters"].get("model_path", {"string_value": None})["string_value"]
        if model_path is None:
            raise ValueError("model_path parameter not set in the model config file")

        offload_folder = model_config["parameters"].get("offload_folder", {"string_value": None})["string_value"]
        if offload_folder is None:
            raise ValueError("offload_folder parameter not set in the model config file")

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.padding_side = "left"

        # initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            offload_folder=offload_folder,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            prompts_np = pb2numpy(request, "prompts")

            prompt = prompts_np[0, 0].decode("utf-8")
            print(f"{prompt=}")

            prompts_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length")

            print(f"{prompts_encoded=}")

            suggestion = self.model.generate(
                inputs=prompts_encoded["input_ids"],
                attention_mask=prompts_encoded["attention_mask"],
                max_new_tokens=20,
                top_k=3,
                penalty_alpha=0.4,
                pad_token_id=50256)

            print(f"{suggestion=}")
            print(f"{suggestion.is_cuda=}")

            suggestion_decoded = self.tokenizer.batch_decode(
                suggestion,
                skip_special_tokens=True)

            print(f"{suggestion_decoded=}")

            completions_np = np.array(suggestion_decoded, dtype="object")
            completions_pb = numpy2pb("completions", completions_np)

            response = pb_utils.InferenceResponse([completions_pb])
            responses.append(response)

        return responses
