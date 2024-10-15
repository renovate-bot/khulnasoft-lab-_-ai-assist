from abc import ABC
from typing import Any, Optional

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from ai_gateway.gitlab_features import GitLabUnitPrimitive

__all__ = [
    "BaseTool",
    "BaseRemoteTool",
]


class BaseTool(ABC, BaseModel, frozen=True):
    name: str
    description: str
    unit_primitive: GitLabUnitPrimitive
    min_required_gl_version: Optional[str] = None
    resource: Optional[str] = None
    example: Optional[str] = None

    def is_compatible(self, gl_version: str) -> bool:
        if not self.min_required_gl_version:
            return True

        if not gl_version:
            return False

        try:
            return Version(self.min_required_gl_version) <= Version(gl_version)
        except InvalidVersion:
            return False


class BaseRemoteTool(BaseTool):
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # By default, we run tools on the Ruby app side
        raise NotImplementedError(
            "Please check the Rails app for an implementation of this tool."
        )
