from dependency_injector import containers, providers

from codesuggestions.auth import GitLabAuthProvider
from codesuggestions.api import middleware

__all__ = [
    "FastApiContainer",
]


class FastApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["__main__"])

    config = providers.Configuration()

    auth_provider = providers.Singleton(
        GitLabAuthProvider,
        base_url=config.auth.gitlab_api_base_url,
    )

    auth_middleware = providers.Factory(
        middleware.MiddlewareAuthentication,
        auth_provider,
        bypass_auth=config.auth.bypass
    )
