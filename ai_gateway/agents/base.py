from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, Tuple, TypeVar

from jinja2 import BaseLoader, Environment
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import Runnable, RunnableBinding
from pydantic import BaseModel

from ai_gateway.auth.user import GitLabUser
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives

__all__ = ["Agent", "BaseAgentRegistry", "BaseAgentConfig", "AgentConfig"]

Input = TypeVar("Input")
Output = TypeVar("Output")

# Agents may operate with unit primitives in various ways.
# Basic agents typically use plain strings as unit primitives.
# More sophisticated agents, like Duo Chat, assign unit primitives to specific tools.
# Creating a generic UnitPrimitiveType enables storage of unit primitives in any desired format.
TypeUnitPrimitive = TypeVar("TypeUnitPrimitive")

jinja_env = Environment(loader=BaseLoader())


def _format_str(content: str, options: dict[str, Any]) -> str:
    return jinja_env.from_string(content).render(options)


class BaseAgentConfig(BaseModel, Generic[TypeUnitPrimitive]):
    name: str
    provider: str
    model: str
    unit_primitives: list[TypeUnitPrimitive]
    prompt_template: dict[str, str]
    stop: Optional[list[str]] = None


class AgentConfig(BaseAgentConfig):
    unit_primitives: list[GitLabUnitPrimitive]


class Agent(RunnableBinding[Input, Output]):
    name: str
    unit_primitives: list[GitLabUnitPrimitive]

    def __init__(
        self, name: str, unit_primitives: list[GitLabUnitPrimitive], chain: Runnable
    ):
        super().__init__(name=name, unit_primitives=unit_primitives, bound=chain)  # type: ignore[call-arg]

    # Assume that the prompt template keys map to roles. Subclasses can
    # override this method to implement more complex logic.
    @staticmethod
    def _prompt_template_to_messages(
        tpl: dict[str, str], options: dict[str, Any]
    ) -> list[Tuple[str, str]]:
        return list(tpl.items())

    @classmethod
    def build_messages(
        cls, prompt_template: dict[str, str], options: dict[str, Any]
    ) -> Sequence[MessageLikeRepresentation]:
        messages = []

        for role, template in cls._prompt_template_to_messages(
            prompt_template, options
        ):
            messages.append((role, _format_str(template, options)))

        return messages


class BaseAgentRegistry(ABC):
    @abstractmethod
    def get(self, agent_id: str, options: Optional[dict[str, Any]] = None) -> Agent:
        pass

    def get_on_behalf(
        self, user: GitLabUser, agent_id: str, options: Optional[dict[str, Any]] = None
    ) -> Agent:
        agent = self.get(agent_id, options)

        if not set(agent.unit_primitives).issubset(user.unit_primitives):
            raise WrongUnitPrimitives

        return agent
