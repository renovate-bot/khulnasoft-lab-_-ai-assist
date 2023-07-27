import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import structlog

from codesuggestions.api.middleware import GitLabUser
from codesuggestions.config import Project

__all__ = [
    "ModelRollout",
    "ModelRolloutBasePlan",
    "ModelRolloutThirdPartyPlan",
    "ModelRolloutWithFallbackPlan",
]

log = structlog.stdlib.get_logger("codesuggestions")


class ModelRollout(str, Enum):
    GITLAB_CODEGEN = "codegen"
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


class ModelRolloutThirdPartyPlan(ModelRolloutBasePlan):
    def __init__(
        self,
        rollout_percentage: int,
        third_party_ai_models: list[ModelRollout],
        f_third_party_ai_default: bool,
        f_limited_access_third_party_ai: dict[int, Project],
    ):
        super().__init__(rollout_percentage)

        self.third_party_models = third_party_ai_models
        self.f_third_party_ai_default = f_third_party_ai_default
        self.f_limited_access_third_party_ai = f_limited_access_third_party_ai

    def _is_third_party_ai_limited_access(self, project_id: Optional[int]) -> bool:
        # Hack: Manually activate third-party AI service
        # for selected testers by project_id
        if project := self.f_limited_access_third_party_ai.get(project_id, None):
            log.info(
                "Redirect request to the third-party model",
                project_id=project.id,
                project_name=project.full_name,
            )
            return True

        return False

    def _resolve_third_party_ai_flag(self, user: GitLabUser) -> bool:
        if is_debug := user.is_debug:
            return is_debug and self.f_third_party_ai_default

        if claims := user.claims:
            return claims.is_third_party_ai_default and self.f_third_party_ai_default

        return False

    def route(self, user: GitLabUser, project_id: Optional[int] = None) -> ModelRollout:
        """
        This function receives a project and returns a model.
        Parameters
        ----------
        user : GitLabUser
          Contains user meta-data

        project_id : int
          The id of the project

        Returns
        -------
        model : Model
          A GitLab model or third-party model. We take a CRC32 checksum of project id and
          perform a mod calculation with 100. If the result is smaller than the rollout
          percentage, projects are equally allocated to one of the third-party models. Since
          this is based on the project id, the allocated model is deterministic. If the result
          is larger than the rollout percentage, the project will use GitLab Codegen model.
        """

        f_third_party_ai_flag = self._resolve_third_party_ai_flag(user)
        is_third_party_ai_limited_access = self._is_third_party_ai_limited_access(
            project_id
        )

        if not (
            project_id and (f_third_party_ai_flag or is_third_party_ai_limited_access)
        ):
            return ModelRollout.GITLAB_CODEGEN

        feature_project = f"third_party:{project_id}"
        checksum = zlib.crc32(feature_project.encode("utf-8"))
        is_included = self._is_project_included(checksum)

        if not (is_included or is_third_party_ai_limited_access):
            return ModelRollout.GITLAB_CODEGEN

        bucket_index = checksum % len(self.third_party_models)
        return self.third_party_models[bucket_index]


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
