from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import yaml
from anthropic import AsyncAnthropic

from ai_gateway.agents.base import Agent
from ai_gateway.models.anthropic import AnthropicChatModel

__all__ = ["BaseAgentRegistry", "LocalAgentRegistry"]


class BaseAgentRegistry(ABC):
    @abstractmethod
    def get(self, usecase: str, key: str) -> Agent:
        pass


class LocalAgentRegistry(BaseAgentRegistry):
    # TODO: Generalize to models from any provider
    def __init__(self, client: AsyncAnthropic):
        self.client = client
        self.base_path = Path(__file__).parent

    @lru_cache
    def get(self, usecase: str, key: str) -> Agent:
        with open(self.base_path / usecase / f"{key}.yml", "r") as f:
            agent_definition = yaml.safe_load(f)

        return Agent(
            name=agent_definition["name"],
            model=AnthropicChatModel.from_model_name(
                agent_definition["model"], client=self.client
            ),
            prompt_templates=agent_definition["prompt_templates"],
        )
