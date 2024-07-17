import functools
import typing

from fastapi import BackgroundTasks, HTTPException, Request, status

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.auth.user import GitLabUser
from ai_gateway.gitlab_features import (
    UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING,
    GitLabUnitPrimitive,
)


def authorize_with_unit_primitive_header():
    """
    Authorize with x-gitlab-unit-primitive header.

    See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/auth.md#use-x-gitlab-unit-primitive-header
    for more information.
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            background_tasks: BackgroundTasks,
            abuse_detector: AbuseDetector,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            await _validate_request(request, background_tasks, abuse_detector)
            return await func(
                request, background_tasks, abuse_detector, *args, **kwargs
            )

        return wrapper

    return decorator


async def _validate_request(
    request: Request, background_tasks: BackgroundTasks, abuse_detector: AbuseDetector
) -> None:
    if X_GITLAB_UNIT_PRIMITIVE not in request.headers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing {X_GITLAB_UNIT_PRIMITIVE} header",
        )

    unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]

    if unit_primitive not in GitLabUnitPrimitive.__members__.values():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown unit primitive header {unit_primitive}",
        )

    unit_primitive = GitLabUnitPrimitive(unit_primitive)

    current_user: GitLabUser = request.user

    if not current_user.can(unit_primitive):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access {unit_primitive}",
        )

    if abuse_detector.should_detect():
        body = await request.body()
        body = body.decode("utf-8", errors="ignore")
        description = UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING.get(unit_primitive, "")
        background_tasks.add_task(abuse_detector.detect, request, body, description)
