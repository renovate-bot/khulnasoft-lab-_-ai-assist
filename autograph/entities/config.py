from typing import List

from pydantic import BaseModel

__all__ = ["AgentConfig", "WorkflowConfig"]


class AgentConfig(BaseModel):
    name: str
    system_prompt: str
    model: str
    goal: str
    temperature: float
    tools: List[str]


class WorkflowConfig(BaseModel):
    name: str
    example_prompt: str
    agents: List[AgentConfig]
    workflow: List[str]
