import zlib
import structlog
from enum import Enum
from typing import Optional

from codesuggestions.api.middleware import GitLabUser
from codesuggestions.config import Project

__all__ = [
    "ModelRollout",
    "ModelRolloutPlan",
]

log = structlog.stdlib.get_logger("codesuggestions")


class ModelRollout(str, Enum):
    GITLAB_CODEGEN = "codegen"
    GOOGLE_TEXT_BISON = "text-bison"
    GOOGLE_CODE_BISON = "code-bison"
    GOOGLE_CODE_GECKO = "code-gecko"


class ModelRolloutPlan:
    def __init__(
        self,
        rollout_percentage: int,
        third_party_ai_models: list[ModelRollout],
        f_third_party_ai_default: bool,
        f_limited_access_third_party_ai: dict[int, Project]
    ):
        self.rollout_percentage = rollout_percentage
        self.third_party_models = third_party_ai_models
        self.f_third_party_ai_default = f_third_party_ai_default
        self.f_limited_access_third_party_ai = f_limited_access_third_party_ai

    def _is_third_party_ai_limited_access(self, project_id: Optional[int]) -> bool:
        # Hack: Manually activate third-party AI service
        # for selected testers by project_id
        if project := self.f_limited_access_third_party_ai.get(project_id, None):
            log.info(
                "Redirect request to the third-party model",
                project_id=project.id, project_name=project.full_name,
            )
            return True

        return False

    def _resolve_third_party_ai_flag(self, user: GitLabUser) -> bool:
        if is_debug := user.is_debug:
            return is_debug and self.f_third_party_ai_default

        if claims := user.claims:
            return claims.is_third_party_ai_default and self.f_third_party_ai_default

        return False

    def route(self, user: GitLabUser, project_id: Optional[int]) -> ModelRollout:
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
        is_third_party_ai_limited_access = self._is_third_party_ai_limited_access(project_id)

        if not (
            project_id
            and (f_third_party_ai_flag or is_third_party_ai_limited_access)
        ):
            return ModelRollout.GITLAB_CODEGEN

        feature_project = f"third_party:{project_id}"
        checksum = zlib.crc32(feature_project.encode("utf-8"))

        # Check if the project is within the rollout percentage
        is_included = checksum % (100 * 1_000) < self.rollout_percentage * 1_000

        if not (
            is_included
            or is_third_party_ai_limited_access
        ):
            return ModelRollout.GITLAB_CODEGEN

        bucket_index = checksum % len(self.third_party_models)
        return self.third_party_models[bucket_index]
