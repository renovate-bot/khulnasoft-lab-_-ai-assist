from http.client import responses
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, WrongUnitPrimitives
from pydantic import BaseModel, RootModel
from starlette.responses import StreamingResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import (
    get_container_application,
    get_internal_event_client,
)
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.prompts import BasePromptRegistry, Prompt
from ai_gateway.prompts.typing import ModelMetadata


class PromptInputs(RootModel):
    root: dict[str, Any]


class PromptRequest(BaseModel):
    inputs: PromptInputs
    stream: Optional[bool] = False
    model_metadata: Optional[ModelMetadata] = None


router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post(
    "/{prompt_id:path}",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
@feature_category(GitLabFeatureCategory.AI_ABSTRACTION_LAYER)
async def prompt(
    request: Request,
    prompt_request: PromptRequest,
    prompt_id: str,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    try:
        prompt = prompt_registry.get_on_behalf(
            current_user, prompt_id, prompt_request.model_metadata
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_id}' not found",
        )
    except WrongUnitPrimitives:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access '{prompt_id}'",
        )

    for unit_primitive in prompt.unit_primitives:
        internal_event_client.track_event(
            f"request_{unit_primitive}",
            category=__name__,
        )

    # We don't use `isinstance` because we don't want to match subclasses
    if not type(prompt) is Prompt:  # pylint: disable=unidiomatic-typecheck
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prompt '{prompt_id}' is not supported",
        )

    try:
        if prompt_request.stream:
            response = prompt.astream(prompt_request.inputs.root)

            async def _handle_stream():
                async for chunk in response:
                    yield chunk.content

            return StreamingResponse(_handle_stream(), media_type="text/event-stream")

        response = await prompt.ainvoke(prompt_request.inputs.root)
        return response.content
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        if hasattr(e, "status_code"):
            status_code = e.status_code
            err_status = f"{status_code}: {responses[status_code]}"
            raise HTTPException(
                status_code=status.HTTP_421_MISDIRECTED_REQUEST,
                # Misdirected Request https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/421
                # To signal to the client the model has produced and error
                # and not the AI Gateway
                detail=err_status,
            )
        raise e
