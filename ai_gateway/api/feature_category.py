import functools
import typing

from fastapi import HTTPException, Request, status
from starlette_context import context

from ai_gateway.cloud_connector import (
    FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)

X_GITLAB_UNIT_PRIMITIVE = "x-gitlab-unit-primitive"

_CATEGORY_CONTEXT_KEY = "meta.feature_category"
_UNIT_PRIMITIVE_CONTEXT_KEY = "meta.unit_primitive"
_UNKNOWN_FEATURE_CATEGORY = "unknown"


def feature_category(name: GitLabFeatureCategory):
    """
    Track a feature category in a single purpose endpoint.

    Example:

    ```
    @feature_category(GitLabFeatureCategory.DUO_CHAT)
    ```
    """
    try:
        GitLabFeatureCategory(name)
    except ValueError:
        raise ValueError(f"Invalid feature category: {name}")

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            context[_CATEGORY_CONTEXT_KEY] = name
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def feature_categories(mapping: dict[GitLabUnitPrimitive, GitLabFeatureCategory]):
    """
    Track feature categories in a multi purpose endpoint.

    It gets the purpose of API call from X-GitLab-Unit-Primitive header,
    identifies the corresponding feature category and stores them in the Starlette context.

    Example:

    ```
    @feature_category({
        GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
        GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    }
    ```
    """
    for category in mapping.values():
        try:
            GitLabFeatureCategory(category)
        except ValueError:
            raise ValueError(f"Invalid feature category: {category}")

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

            try:
                feature_category = mapping[unit_primitive]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This endpoint cannot be used for {unit_primitive} purpose",
                )

            context[_CATEGORY_CONTEXT_KEY] = feature_category
            context[_UNIT_PRIMITIVE_CONTEXT_KEY] = unit_primitive
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def track_metadata(request_param: str, mapping: dict[str, GitLabUnitPrimitive]):
    """
    Track feature category and unit primitive from request path.

    Example:

    ```
    @track_metadata(
         "chat_invokable",
         mapping={
             "explain_vulnerability": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
             "troubleshoot_job": GitLabUnitPrimitive.TROUBLESHOOT_JOB,
         }
    )
    ```
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            request_param_val = request.path_params[request_param]

            if request_param_val in mapping:
                unit_primitive = mapping[request_param_val]
                feature_category = FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS[
                    unit_primitive
                ]

                context[_CATEGORY_CONTEXT_KEY] = feature_category.value
                context[_UNIT_PRIMITIVE_CONTEXT_KEY] = unit_primitive.value

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def current_feature_category() -> str:
    """
    Get the feature category set to the current request context.
    """
    if context.exists():
        feature_category = context.get(_CATEGORY_CONTEXT_KEY, _UNKNOWN_FEATURE_CATEGORY)

        if isinstance(feature_category, GitLabFeatureCategory):
            return feature_category.value

        return feature_category

    return _UNKNOWN_FEATURE_CATEGORY
