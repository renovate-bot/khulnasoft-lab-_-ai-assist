from typing import Optional

from codesuggestions.models import TextGenBaseModel, TextGenModelOutput

__all__ = [
    "FakeGitLabCodeGenModel",
    "FakePalmTextGenModel",
]


class FakeGitLabCodeGenModel(TextGenBaseModel):
    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.0,
        max_output_tokens: int = 0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> Optional[TextGenModelOutput]:
        return TextGenModelOutput(text="fake code suggestion from GitLab Codegen")


class FakePalmTextGenModel(TextGenBaseModel):
    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.0,
        max_output_tokens: int = 0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> Optional[TextGenModelOutput]:
        return TextGenModelOutput(text="fake code suggestion from PaLM Text")
