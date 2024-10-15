import functools
from typing import Any, Awaitable, Callable

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends
from fastapi_health import health

from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_completions_vertex_legacy_provider,
    get_code_suggestions_generations_anthropic_factory_provider,
)
from ai_gateway.code_suggestions import CodeCompletionsLegacy, CodeGenerations
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
    completions_legacy_vertex_factory: Factory[CodeCompletionsLegacy] = Depends(
        get_code_suggestions_completions_vertex_legacy_provider
    ),
) -> bool:
    code_completions = completions_legacy_vertex_factory()
    await code_completions.execute(
        prefix="def hello_world():",
        suffix="",
        file_name="monitoring.py",
        editor_lang="python",
    )
    return True


@single_validation(KindModelProvider.ANTHROPIC)
async def validate_anthropic_available(
    generations_anthropic_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_anthropic_factory_provider
    ),
) -> bool:
    code_generations = generations_anthropic_factory(
        model__name=KindAnthropicModel.CLAUDE_3_HAIKU.value,
        model__stop_sequences=["</new_code>"],
    )

    # The generation prompt is currently built in rails, so include a minimal one
    # here to replace that
    await code_generations.execute(
        prefix="\n\nHuman: Complete this code: def hello_world():\n\nAssistant:",
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
