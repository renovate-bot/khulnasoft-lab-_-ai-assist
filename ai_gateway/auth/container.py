from dependency_injector import containers, providers
from gitlab_cloud_connector import TokenAuthority

from ai_gateway.auth.glgo import GlgoAuthority

__all__ = ["ContainerSelfSignedJwt"]


class ContainerSelfSignedJwt(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    token_authority: TokenAuthority = providers.Factory(
        TokenAuthority,
        signing_key=config.self_signed_jwt.signing_key,
    )

    glgo_authority = providers.Singleton(
        GlgoAuthority,
        signing_key=config.self_signed_jwt.signing_key,
        glgo_base_url=config.glgo_base_url,
    )
