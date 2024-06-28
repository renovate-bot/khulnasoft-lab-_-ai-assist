from dependency_injector import containers, providers

from ai_gateway.auth.self_signed_jwt import TokenAuthority

__all__ = ["ContainerSelfSignedJwt"]


class ContainerSelfSignedJwt(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    token_authority = providers.Factory(
        TokenAuthority,
        signing_key=config.self_signed_jwt.signing_key,
    )
