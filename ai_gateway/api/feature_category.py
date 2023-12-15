import functools
import typing

from starlette_context import context

# TODO: Validate that these categories exist in https://gitlab.com/gitlab-com/www-gitlab-com/raw/master/data/stages.yml.
_FEATURE_CATEGORIES = [
    "code_suggestions",
    "duo_chat",
]


def feature_category(name: str):
    if name not in _FEATURE_CATEGORIES:
        raise ValueError(f"Invalid feature category: {name}")

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            context["meta.feature_category"] = name
            return await func(*args, **kwargs)

        return wrapper

    return decorator
