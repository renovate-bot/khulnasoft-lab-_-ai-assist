from dependency_injector import containers, providers

__all__ = [
    "ContainerChat",
]


class ContainerChat(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    # We need to resolve the model based on model name provided in request payload
    # Hence, `models.anthropic_claude` is only partially applied here.
    anthropic_claude_factory = providers.Factory(models.anthropic_claude)
