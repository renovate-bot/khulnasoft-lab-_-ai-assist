import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import structlog

from ai_gateway.api.middleware import GitLabUser

__all__ = [
    "ModelRollout",
    "ModelRolloutBasePlan",
    "ModelRolloutWithFallbackPlan",
]

log = structlog.stdlib.get_logger("codesuggestions")


class ModelRollout(str, Enum):
    GOOGLE_TEXT_BISON = "text-bison"
    GOOGLE_CODE_BISON = "code-bison"
    GOOGLE_CODE_GECKO = "code-gecko"


class ModelRolloutBasePlan(ABC):
    def __init__(self, rollout_percentage: int):
        self.rollout_percentage = rollout_percentage

    def _is_project_included(self, checksum: int) -> bool:
        """
        Check if the project checksum is within the rollout percentage
        :return bool
        """

        return checksum % (100 * 1_000) < self.rollout_percentage * 1_000

    @abstractmethod
    def route(self, user: GitLabUser, project_id: Optional[int] = None) -> ModelRollout:
        pass


class ModelRolloutWithFallbackPlan(ModelRolloutBasePlan):
    def __init__(
        self,
        rollout_percentage: int,
        primary_model: ModelRollout,
        fallback_model: ModelRollout,
    ):
        super().__init__(rollout_percentage)

        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def route(self, user: GitLabUser, project_id: Optional[int] = None) -> ModelRollout:
        project_id = project_id if project_id else -1

        feature_project = f"with_fallback:{project_id}"
        checksum = zlib.crc32(feature_project.encode("utf-8"))
        is_included = self._is_project_included(checksum)

        return self.primary_model if is_included else self.fallback_model
