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
    MODEL_NAME = "ensemble"

    # Number of tokens to generate
    REQUEST_OUTPUT_LEN = 16

    # Model hyperparameters
    MODEL_TEMPERATURE = .2
    MODEL_REPETITION_PENALTY = 1
    MODEL_TOP_K = 0
    MODEL_TOP_P = .98
    MODEL_PAD_ID = 50256

    def __init__(self, grpc_client: triton_grpc_util.InferenceServerClient, timeout: int = 10):
        self.client = grpc_client
        self.timeout = timeout

    # noinspection PyMethodMayBeStatic
    def _trim_prompt_max_len(self, prompt: str) -> str:
        return prompt[-Codegen.MAX_MODEL_LEN:]

    def _model_input(self, prompt: str) -> list:
        prompt_np = np.array([[prompt]], dtype=object)
        request_output_len_np = np.array([[self.REQUEST_OUTPUT_LEN]]).astype(np.uint32)
        temperature_np = np.array([[self.MODEL_TEMPERATURE]]).astype(np.float32)
        repetition_penalty_np = np.array([[self.MODEL_REPETITION_PENALTY]]).astype(np.float32)
        top_k_np = np.array([[self.MODEL_TOP_K]]).astype(np.uint32)
        top_p_np = np.array([[self.MODEL_TOP_P]]).astype(np.float32)
        start_id_np = np.array([[self.MODEL_PAD_ID]]).astype(np.uint32)
        end_id_np = np.array([[self.MODEL_PAD_ID]]).astype(np.uint32)
        random_seed_np = np.random.randint(0, 2 ** 31 - 1, (1, 1), dtype=np.int32)

        return [
            grpc_input_from_np("prompt", prompt_np),
            grpc_input_from_np("request_output_len", request_output_len_np),
            grpc_input_from_np("temperature", temperature_np),
            grpc_input_from_np("repetition_penalty", repetition_penalty_np),
            grpc_input_from_np("runtime_top_k", top_k_np),
            grpc_input_from_np("runtime_top_p", top_p_np),
            grpc_input_from_np("start_id", start_id_np),
            grpc_input_from_np("end_id", end_id_np),
            grpc_input_from_np("random_seed", random_seed_np),
            grpc_input_from_np("is_return_log_probs", np.array([[1]]).astype(bool)),
        ]

    # noinspection PyMethodMayBeStatic
    def _model_outputs(self) -> list:
        return [
            grpc_requested_output("completions"),
            grpc_requested_output("sequence_length"),
            grpc_requested_output("output_log_probs"),
            grpc_requested_output("cum_log_probs"),
            grpc_requested_output("output_ids"),
        ]

    def __call__(self, prompt: str) -> str:
        prompt = self._trim_prompt_max_len(prompt)

        inputs_model = self._model_input(prompt)
        outputs_model = self._model_outputs()

        response = self.client.infer(
            Codegen.MODEL_NAME,
            inputs_model,
            request_id=uuid.uuid4().hex,
            outputs=outputs_model,
        )

        completion = (
            response.as_numpy("completions")[0]
            .decode("utf-8")
        )

        return completion
