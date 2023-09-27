from ai_gateway.code_suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineOutput,
)

__all__ = ["CodeCompletions"]


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
