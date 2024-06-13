import functools
import typing

from fastapi import HTTPException, Request, status

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.auth.user import GitLabUser


def authorize_with_unit_primitive_header():
    """
    Authorize with x-gitlab-unit-primitive header.
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            try:
                unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing {X_GITLAB_UNIT_PRIMITIVE} header",
                )

            current_user: GitLabUser = request.user

            if not current_user.can(unit_primitive):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unauthorized to access {unit_primitive}",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
