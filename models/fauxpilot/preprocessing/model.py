import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

PAD_ID = 50256


class TritonPythonModel:
    def __init__(self):
        self.tokenizer = None

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args['model_config'])

        # Parse model output configs and convert Triton types to numpy types
        output_names = ["INPUT_IDS", "REQUEST_INPUT_LEN"]
        for input_name in output_names:
            setattr(self,
                    input_name.lower() + "_dtype",
                    pb_utils.triton_string_to_numpy(pb_utils.get_output_config_by_name(
                        model_config, input_name)['data_type'])
                    )

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            prompt = pb_utils.get_input_tensor_by_name(request, 'PROMPT').as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(request, 'REQUEST_OUTPUT_LEN').as_numpy()

            # Preprocessing input data
            input_ids = [torch.IntTensor(self.tokenizer.encode(s[0].decode())) for s in prompt]
            input_lengths = torch.IntTensor([[len(ids)] for ids in input_ids])
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)

            # Create output tensors
            input_id_tensor = pb_utils.Tensor(
                'INPUT_IDS',
                np.array(input_ids).astype(self.input_ids_dtype)
            )
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                np.array(input_lengths).astype(self.request_input_len_dtype)
            )
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN',
                request_output_len
            )

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor,
                request_input_len_tensor,
                request_output_len_tensor,
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
