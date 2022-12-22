from codesuggestions.models import BaseModel

__all__ = [
    "CodeSuggestionsUseCase",
]


class CodeSuggestionsUseCase:
    def __init__(self, model: BaseModel):
        self.model = model

    # noinspection PyMethodMayBeStatic
    def _preprocess_prompt(self, prompt: str, *funcs) -> str:
        for func in funcs:
            prompt = func(prompt)
        return prompt

    def __call__(self, prompt: str) -> str:
        completion = self.model(prompt)

        return completion
