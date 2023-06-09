from typing import Optional

from vertexai.preview.language_models import TextGenerationModel

from codesuggestions.models import TextGenBaseModel, TextGenModelOutput
from codesuggestions.models.instrumentators import TextGenModelInstrumentator

__all__ = [
    "PalmTextGenModel",
]


class PalmTextGenModel(TextGenBaseModel):
    # Max number of tokens the model can handle
    MAX_MODEL_LEN = 2048

    def __init__(self, model_name: str, timeout: int = 30):
        self.model_name = model_name

        # Access the endpoint object directly to adjust timeout
        self.endpoint = TextGenerationModel.from_pretrained(model_name)._endpoint
        self.timeout = timeout
        self.instrumentator = TextGenModelInstrumentator("vertex-ai", model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_decode_steps: int = 32,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Optional[TextGenModelOutput]:
        instances = [{"content": prompt}]
        prediction_parameters = {
            "temperature": temperature,
            "maxDecodeSteps": max_decode_steps,
            "topK": top_k,
            "topP": top_p,
        }

        with self.instrumentator.watch(prompt):
            response = self.endpoint.predict(
                instances=instances,
                parameters=prediction_parameters,
                timeout=self.timeout,
            )

        if len(response.predictions) > 0:
            prediction = response.predictions.pop()
            return TextGenModelOutput(text=prediction["content"])

        return None
