from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, TypeVar

from langchain_core.runnables import Runnable, RunnableConfig

__all__ = ["Agent", "BaseAgentRegistry"]

Input = TypeVar("Input")
Output = TypeVar("Output")


class Agent(Runnable[Input, Output]):
    def __init__(self, name: str, chain: Runnable):
        self.name = name
        self.chain = chain

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        raise NotImplementedError

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        raise NotImplementedError


class BaseAgentRegistry(ABC):
    @abstractmethod
    def get(self, use_case: str, agent_type: str, **kwargs: Optional[Any]) -> Any:
        pass
