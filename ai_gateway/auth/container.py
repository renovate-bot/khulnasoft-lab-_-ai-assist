from dependency_injector import containers, providers
from gitlab_cloud_connector import TokenAuthority

__all__ = ["ContainerSelfSignedJwt"]


class ContainerSelfSignedJwt(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    token_authority: TokenAuthority = providers.Factory(
        TokenAuthority,
        signing_key=config.self_signed_jwt.signing_key,
    )
