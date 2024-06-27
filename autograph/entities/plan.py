from enum import Enum
from typing import List

from pydantic import BaseModel, Field

__all__ = ["Plan", "TaskStatusEnum", "TaskStatusInput", "Task"]


class TaskStatusEnum(str, Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


class TaskStatusInput(BaseModel):
    status: TaskStatusEnum = Field(description="The current status of the task")


class Task(BaseModel):
    description: str = Field(description="A description of what the task is")
    status: str = Field(
        description=f"""The status of the task.
                        The status can be {", ".join([status.value for status in TaskStatusEnum])}"""
    )


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Task] = Field(
        description="different steps to follow, should be in sorted order"
    )
