from codesuggestions.suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineGenerations,
    ModelEngineOutput,
)

__all__ = ["CodeCompletions", "CodeGenerations"]


class CodeCompletions:
    def __init__(self, engine: ModelEngineCompletions):
        self.engine = engine

    def __call__(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        language_identifier: str,
    ) -> ModelEngineOutput:
        return self.engine.generate(prefix, suffix, file_name, language_identifier)


class CodeGenerations:
    def __init__(self, engine: ModelEngineGenerations):
        self.engine = engine

    def __call__(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        language_identifier: str,
        prompt_input: str = None,
    ) -> ModelEngineOutput:
        return self.engine.generate(
            prefix, suffix, file_name, language_identifier, prompt_input=prompt_input
        )
