from typing import Optional, Any, NamedTuple

from starlette_context import context

from codesuggestions.suggestions.processing import ModelEngineBase

__all__ = [
    "CodeCompletionsInternalUseCase",
]


class CodeCompletionsInternal(NamedTuple):
    class Model(NamedTuple):
        # TODO: replace with enum values
        engine: str
        name: str

    text: str
    model: Model
    finish_reason: str = "length"


class CodeCompletionsInternalUseCase:
    def __init__(self, engine: ModelEngineBase):
        self.engine = engine

    async def __call__(
        self,
        prefix: str,
        suffix: str,
        file_name: Optional[str] = None,
        **kwargs: Any
    ) -> CodeCompletionsInternal:
        file_name = file_name if file_name else ""

        completion = await self.engine.generate_completion(
            prefix,
            suffix,
            file_name,
            **kwargs,
        )

        return CodeCompletionsInternal(
            text=completion.text,
            model=CodeCompletionsInternal.Model(
                # TODO: return props from the target engine instead of using glob var
                engine=context.get("model_engine", ""),
                name=context.get("model_name", "")
            )
        )
