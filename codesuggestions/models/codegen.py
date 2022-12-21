import uuid

import numpy as np
import tritonclient.grpc as triton_grpc_util

from codesuggestions.models.base import BaseModel, grpc_input_from_np, grpc_requested_output

__all__ = [
    "Codegen",
]


class Codegen(BaseModel):
    # Max number of tokens the model can handle
    MAX_MODEL_LEN = 2048

    # Model name specified in config.pbtxt
    MODEL_NAME = "codegen"

    def __init__(self, grpc_client: triton_grpc_util.InferenceServerClient, timeout: int = 10):
        self.client = grpc_client
        self.timeout = timeout

    # noinspection PyMethodMayBeStatic
    def _trim_prompt_max_len(self, prompt: str) -> str:
        return prompt[-Codegen.MAX_MODEL_LEN:]

    def __call__(self, prompt: str) -> str:
        prompt = self._trim_prompt_max_len(prompt)
        prompts_np = np.array([[prompt]], dtype=object)

        inputs_model = [
            grpc_input_from_np("prompts", prompts_np)
        ]
        outputs_model = [
            grpc_requested_output("completions"),
        ]

        response = self.client.infer(
            Codegen.MODEL_NAME,
            inputs_model,
            request_id=uuid.uuid4().hex,
            client_timeout=self.timeout,
            outputs=outputs_model,
        )

        completions = (
            response.as_numpy("completions")[0]
            .decode("utf-8")
        )

        return completions
