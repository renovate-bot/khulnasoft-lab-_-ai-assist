from typing import Annotated

from fastapi import APIRouter, Depends, Request
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v3.code.completions import code_suggestions as v3_code_suggestions
from ai_gateway.api.v3.code.typing import CompletionRequest
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.prompts import BasePromptRegistry

__all__ = [
    "router",
]

router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post("/suggestions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def suggestions(
    request: Request,
    payload: CompletionRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    return await v3_code_suggestions(
        request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
    )
