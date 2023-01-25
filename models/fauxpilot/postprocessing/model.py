import json
import numpy as np
import triton_python_backend_utils as pb_utils

from transformers import AutoTokenizer


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

        # Convert Triton types to numpy types
        output_names = ["OUTPUT"]
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
            tokens_batch = pb_utils.get_input_tensor_by_name(request, 'TOKENS_BATCH').as_numpy()
            req_input_len_batch = pb_utils.get_input_tensor_by_name(request, 'REQUEST_INPUT_LEN').as_numpy()

            # Postprocessing output data
            outputs = []
            for beam_tokens, req_input_len in zip(tokens_batch, req_input_len_batch):
                for tokens in beam_tokens:
                    tokens = tokens[req_input_len[0]:]
                    output = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    outputs.append(output.encode('utf8'))

            # Create output tensors
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
