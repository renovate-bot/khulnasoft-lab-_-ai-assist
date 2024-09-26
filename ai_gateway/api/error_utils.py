import functools

from fastapi import HTTPException, status
from pydantic import ValidationError


def capture_validation_errors():
    def decorator(f):
        @functools.wraps(f)
        async def wrap(*args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except ValidationError as ex:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(ex.errors()),
                )

        return wrap

    return decorator
