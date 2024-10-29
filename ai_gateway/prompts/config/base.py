from pydantic import BaseModel, ConfigDict

from ai_gateway.cloud_connector import GitLabUnitPrimitive
from ai_gateway.prompts.config.models import TypeModelParams

__all__ = ["PromptConfig", "ModelConfig"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: TypeModelParams


class PromptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop: list[str] | None = None
    # NOTE: In langchain, some providers accept the timeout when initializing the client. However, support
    # and naming is inconsistent between them. Therefore, we bind the timeout to the prompt instead.
    # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1035#note_2020952732
    timeout: float | None = None
    vertex_location: str | None = None


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_template: dict[str, str]
    params: PromptParams | None = None
