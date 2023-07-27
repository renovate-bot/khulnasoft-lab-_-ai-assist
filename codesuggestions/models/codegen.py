import uuid

import numpy as np
import tritonclient.grpc as triton_grpc_util
from pydantic import BaseModel

from codesuggestions.instrumentators.base import TextGenModelInstrumentator
from codesuggestions.models.base import (
    TextGenBaseModel,
    TextGenModelOutput,
    grpc_input_from_np,
    grpc_requested_output,
)

__all__ = [
    "GitLabCodeGen",
]


class GitLabCodeGenModelInput(BaseModel):
    prompt: str
    request_output_len: int
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float
    start_id_np: int = 50256
    end_id_np: int = 50256


class GitLabCodeGen(TextGenBaseModel):
    # Max number of tokens the model can handle
    MAX_MODEL_LEN = 2048

    # Model name specified in config.pbtxt
    MODEL_NAME = "ensemble"

    ENGINE_NAME = "codegen"

    def __init__(
        self, grpc_client: triton_grpc_util.InferenceServerClient, timeout: int = 30
    ):
        self.client = grpc_client
        self.timeout = timeout
        self.instrumentator = TextGenModelInstrumentator(
            GitLabCodeGen.ENGINE_NAME, GitLabCodeGen.MODEL_NAME
        )

    def _model_inputs(self, model_input: GitLabCodeGenModelInput) -> list:
        prompt_np = np.array([[model_input.prompt]], dtype=object)
        request_output_len_np = np.array([[model_input.request_output_len]]).astype(
            np.uint32
        )
        temperature_np = np.array([[model_input.temperature]]).astype(np.float32)
        repetition_penalty_np = np.array([[model_input.repetition_penalty]]).astype(
            np.float32
        )
        top_k_np = np.array([[model_input.top_k]]).astype(np.uint32)
        top_p_np = np.array([[model_input.top_p]]).astype(np.float32)
        start_id_np = np.array([[model_input.start_id_np]]).astype(np.uint32)
        end_id_np = np.array([[model_input.end_id_np]]).astype(np.uint32)
        random_seed_np = np.random.randint(0, 2**31 - 1, (1, 1), dtype=np.int32)

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

    @property
    def model_name(self) -> str:
        return GitLabCodeGen.MODEL_NAME

    @property
    def model_engine(self) -> str:
        return GitLabCodeGen.ENGINE_NAME

    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.2,
        max_output_tokens: int = 32,
        top_p: float = 0.98,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> TextGenModelOutput:
        inputs_model = self._model_inputs(
            GitLabCodeGenModelInput(
                prompt=prompt,
                temperature=temperature,
                request_output_len=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ),
        )
        outputs_model = self._model_outputs()

        with self.instrumentator.watch(prompt):
            response = self.client.infer(
                GitLabCodeGen.MODEL_NAME,
                inputs_model,
                request_id=uuid.uuid4().hex,
                outputs=outputs_model,
                client_timeout=self.timeout,
            )

        completion = response.as_numpy("completions")[0].decode("utf-8")

        return TextGenModelOutput(
            text=completion,
        )
