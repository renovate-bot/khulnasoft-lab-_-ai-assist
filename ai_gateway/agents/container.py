from pathlib import Path

from dependency_injector import containers, providers

from ai_gateway.agents.chat import ReActAgent
from ai_gateway.agents.registry import Key, LocalAgentRegistry

__all__ = [
    "ContainerAgents",
]


class ContainerAgents(containers.DeclarativeContainer):
    agent_registry = providers.Singleton(
        LocalAgentRegistry.from_local_yaml,
        data={
            Key(use_case="chat", type="react"): (
                Path(__file__).parent / "chat" / "react.yml",
                ReActAgent,
            )
        },
    )
