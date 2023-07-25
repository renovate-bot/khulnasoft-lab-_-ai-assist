from codesuggestions.suggestions.processing import ModelEngineBase, ModelEngineOutput

__all__ = ["CodeSuggestions"]


class CodeSuggestions:
    def __init__(self, engine: ModelEngineBase):
        self.engine = engine

    def __call__(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
    ) -> ModelEngineOutput:
        return self.engine.generate_completion(prefix, suffix, file_name)
