import functools
import typing

from starlette_context import context

# TODO: Validate that these categories exist in https://gitlab.com/gitlab-com/www-gitlab-com/raw/master/data/stages.yml.
_FEATURE_CATEGORIES = [
    "code_suggestions",
    "duo_chat",
]

_CONTEXT_KEY = "meta.feature_category"
_UNKNOWN_FEATURE_CATEGORY = "unknown"


def feature_category(name: str):
    if name not in _FEATURE_CATEGORIES:
        raise ValueError(f"Invalid feature category: {name}")

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            context[_CONTEXT_KEY] = name
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_feature_category() -> str:
    if context.exists():
        return context.get(_CONTEXT_KEY, _UNKNOWN_FEATURE_CATEGORY)
    else:
        return _UNKNOWN_FEATURE_CATEGORY
