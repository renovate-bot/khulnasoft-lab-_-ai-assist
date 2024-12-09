"""
This module allows you to resolve the dependencies with Python async/coroutines.

Do NOT use FastAPI's `Depends` and Dependency Injector's Wiring (a.k.a `Provide`/`@inject`) in the following way:

```
async def chat(
    ...
    some_factory: Factory[SomeModel] = Depends(Provide[ContainerApplication.some_factory.provider]),
    ...
```

This is [an example usage](https://python-dependency-injector.ets-labs.org/examples/fastapi.html) provided by Dependency Injector.
However, since `Provide` object is not async/coroutine compatible, FastAPI runs a new thread for resolving the dependency.

See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/606 for more information.
"""

from dependency_injector.wiring import Provide, inject

from ai_gateway.container import ContainerApplication


@inject
def get_container_application(
    container: ContainerApplication = Provide[ContainerApplication],
):
    return container


async def get_chat_anthropic_claude_factory_provider():
    yield get_container_application().chat.anthropic_claude_factory


async def get_search_factory_provider():
    yield get_container_application().searches.search_provider


async def get_x_ray_anthropic_claude():
    yield get_container_application().x_ray.anthropic_claude()


async def get_code_suggestions_completions_vertex_legacy_provider():
    yield get_container_application().code_suggestions.completions.vertex_legacy


async def get_code_suggestions_completions_anthropic_provider():
    yield get_container_application().code_suggestions.completions.anthropic


async def get_code_suggestions_completions_litellm_factory_provider():
    yield get_container_application().code_suggestions.completions.litellm_factory


async def get_code_suggestions_completions_litellm_vertex_codestral_factory_provider():
    yield get_container_application().code_suggestions.completions.litellm_vertex_codestral_factory


async def get_code_suggestions_completions_agent_factory_provider():
    yield get_container_application().code_suggestions.completions.agent_factory


async def get_snowplow_instrumentator():
    yield get_container_application().snowplow.instrumentator()


async def get_code_suggestions_generations_vertex_provider():
    yield get_container_application().code_suggestions.generations.vertex


async def get_code_suggestions_generations_anthropic_factory_provider():
    yield get_container_application().code_suggestions.generations.anthropic_factory


async def get_code_suggestions_generations_anthropic_chat_factory_provider():
    yield get_container_application().code_suggestions.generations.anthropic_chat_factory


async def get_code_suggestions_generations_litellm_factory_provider():
    yield get_container_application().code_suggestions.generations.litellm_factory


async def get_code_suggestions_generations_agent_factory_provider():
    yield get_container_application().code_suggestions.generations.agent_factory


async def get_chat_litellm_factory_provider():
    yield get_container_application().chat.litellm_factory


@inject
async def get_anthropic_proxy_client(
    anthropic_proxy_client=Provide[
        ContainerApplication.pkg_models.anthropic_proxy_client
    ],
):
    return anthropic_proxy_client


@inject
async def get_vertex_ai_proxy_client(
    vertex_ai_proxy_client=Provide[
        ContainerApplication.pkg_models.vertex_ai_proxy_client
    ],
):
    return vertex_ai_proxy_client


async def get_internal_event_client():
    return get_container_application().internal_event.client()


@inject
async def get_abuse_detector(
    abuse_detector=Provide[ContainerApplication.abuse_detection.abuse_detector],
):
    return abuse_detector


@inject
async def get_token_authority(
    token_authority=Provide[ContainerApplication.self_signed_jwt.token_authority],
):
    return token_authority


@inject
async def get_glgo_authority(
    glgo_authority=Provide[ContainerApplication.self_signed_jwt.glgo_authority],
):
    return glgo_authority


@inject
async def get_amazon_q_client_factory(
    amazon_q_client_factory=Provide[
        ContainerApplication.integrations.amazon_q_client_factory
    ],
):
    return amazon_q_client_factory
