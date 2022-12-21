from codesuggestions.models import BaseModel

__all__ = [
    "CodeSuggestionsUseCase",
]


class CodeSuggestionsUseCase:
    def __init__(self, model: BaseModel):
        self.model = model

    # noinspection PyMethodMayBeStatic
    def _rstrip_prompt(self, prompt: str) -> str:
        return prompt.rstrip()

    # noinspection PyMethodMayBeStatic
    def _preprocess_prompt(self, prompt: str, *funcs) -> str:
        for func in funcs:
            prompt = func(prompt)
        return prompt

    def __call__(self, prompt: str) -> str:
        prompt = self._preprocess_prompt(
            prompt,
            self._rstrip_prompt,
        )

        completion = self.model(prompt)

        return completion
