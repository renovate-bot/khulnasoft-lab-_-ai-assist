import functools
import typing

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from ai_gateway.gitlab_features import GitLabUnitPrimitive

__all__ = ["ChatInvokable", "authorize_with_unit_primitive"]


class ChatInvokable(BaseModel):
    name: str
    unit_primitive: GitLabUnitPrimitive


def authorize_with_unit_primitive(
    request_param: str, *, chat_invokables: list[ChatInvokable]
):
    def decorator(func: typing.Callable) -> typing.Callable:
        chat_invokable_by_name = {ci.name: ci for ci in chat_invokables}

        @functools.wraps(func)
        async def wrapper(
            request: Request, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            request_param_val = request.path_params[request_param]

            chat_invokable = chat_invokable_by_name.get(request_param_val, None)
            if not chat_invokable:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
                )

            current_user = request.user
            unit_primitive = chat_invokable.unit_primitive
            if not current_user.can(unit_primitive):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unauthorized to access {unit_primitive}",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
