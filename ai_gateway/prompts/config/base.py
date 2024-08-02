from typing import Generic, TypeVar

from pydantic import BaseModel

from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.prompts.config.models import TypeModelParams

__all__ = ["BasePromptConfig", "PromptConfig", "ModelConfig"]


# Prompts may operate with unit primitives in various ways.
# Basic prompts typically use plain strings as unit primitives.
# More sophisticated prompts, like Duo Chat, assign unit primitives to specific tools.
# Creating a generic UnitPrimitiveType enables storage of unit primitives in any desired format.
TypeUnitPrimitive = TypeVar("TypeUnitPrimitive")


class ModelConfig(BaseModel):
    name: str
    params: TypeModelParams


class PromptParams(BaseModel):
    stop: list[str] | None = None
    # NOTE: In langchain, some providers accept the timeout when initializing the client. However, support
    # and naming is inconsistent between them. Therefore, we bind the timeout to the prompt instead.
    # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1035#note_2020952732
    timeout: float | None = None


class BasePromptConfig(BaseModel, Generic[TypeUnitPrimitive]):
    name: str
    model: ModelConfig
    unit_primitives: list[TypeUnitPrimitive]
    prompt_template: dict[str, str]
    params: PromptParams | None = None


class PromptConfig(BasePromptConfig):
    unit_primitives: list[GitLabUnitPrimitive]
