from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import yaml

from ai_gateway.agents import Agent

__all__ = ["BaseAgentRegistry", "LocalAgentRegistry"]


class BaseAgentRegistry(ABC):
    @abstractmethod
    def get(self, usecase: str, key: str) -> Agent:
        pass


class LocalAgentRegistry(BaseAgentRegistry):
    def __init__(self):
        self.base_path = Path(__file__).parent

    @lru_cache
    def get(self, usecase: str, key: str) -> Agent:
        with open(self.base_path / usecase / f"{key}.yml", "r") as f:
            agent_definition = yaml.safe_load(f)

        return Agent(
            name=agent_definition["name"],
            prompt_templates=agent_definition["prompt_templates"],
        )
