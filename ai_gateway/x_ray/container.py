from dependency_injector import containers, providers

from ai_gateway.models import KindAnthropicModel

__all__ = [
    "ContainerXRay",
]


class ContainerXRay(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    anthropic_claude = providers.Factory(
        providers.Factory(models.anthropic_claude, name=KindAnthropicModel.CLAUDE_2_0),
        stop_sequences=["</new_code>", "\n\nHuman:"],
    )
