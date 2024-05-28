import functools
from typing import Any, Awaitable, Callable

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends
from fastapi_health import health

from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_generations_anthropic_factory_provider,
    get_code_suggestions_generations_vertex_provider,
)
from ai_gateway.code_suggestions import CodeGenerations
from ai_gateway.models import KindAnthropicModel, KindModelProvider

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)

# Avoid calling out to the models multiple times from this public, unauthenticated endpoint.
# this is not threadsafe, but that should be fine, we aren't issuing multiple of
# these calls in parallel. When the instance is marked as ready, we won't be modifying
# the list anymore.
validated: set[KindModelProvider] = set()


def single_validation(
    key: KindModelProvider,
):
    def _decorator(
        func: Callable[[Any], Awaitable[bool]]
    ) -> Callable[[Any, Any], Awaitable[bool]]:

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs) -> bool:
            if key in validated:
                return True

            result = await func(*args, **kwargs)
            validated.add(key)

            return result

        return _wrapper

    return _decorator


@single_validation(KindModelProvider.VERTEX_AI)
async def validate_vertex_available(
    generations_vertex_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_vertex_provider
    ),
) -> bool:
    code_suggestions = generations_vertex_factory()
    await code_suggestions.execute(
        prefix="def hello_world():",
        file_name="monitoring.py",
        editor_lang="python",
        model_provider=KindModelProvider.VERTEX_AI.value,
    )
    return True


@single_validation(KindModelProvider.ANTHROPIC)
async def validate_anthropic_available(
    generations_anthropic_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_anthropic_factory_provider
    ),
) -> bool:
    code_generations = generations_anthropic_factory(
        model__name=KindAnthropicModel.CLAUDE_2_0.value,
        model__stop_sequences=["</new_code>"],
    )
    await code_generations.execute(
        prefix="def hello_world():",
        file_name="monitoring.py",
        editor_lang="python",
        model_provider=KindModelProvider.ANTHROPIC.value,
    )

    return True


router.add_api_route("/healthz", health([]))
router.add_api_route(
    "/ready",
    health(
        [
            validate_vertex_available,
            validate_anthropic_available,
        ]
    ),
)
