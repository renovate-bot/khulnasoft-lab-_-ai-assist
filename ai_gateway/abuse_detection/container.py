from dependency_injector import containers, providers

from ai_gateway.abuse_detection.detector import AbuseDetector
from ai_gateway.models.anthropic import KindAnthropicModel


class ContainerAbuseDetection(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)
    models = providers.DependenciesContainer()

    abuse_detector = providers.Singleton(
        AbuseDetector,
        enabled=config.enabled,
        sampling_rate=config.sampling_rate,
        model=providers.Factory(
            models.anthropic_claude_chat, KindAnthropicModel.CLAUDE_3_HAIKU
        ),
    )
