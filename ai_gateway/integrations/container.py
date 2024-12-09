from dependency_injector import containers, providers

from ai_gateway.auth.glgo import GlgoAuthority
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory

__all__ = ["ContainerIntegrations"]


class ContainerIntegrations(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    glgo_authority = providers.Singleton(
        GlgoAuthority,
        signing_key=config.self_signed_jwt.signing_key,
        glgo_base_url=config.glgo_base_url,
    )

    amazon_q_client_factory = providers.Singleton(
        AmazonQClientFactory,
        glgo_authority=glgo_authority,
        endpoint_url=config.amazon_q.endpoint_url,
        region=config.amazon_q.region,
    )
