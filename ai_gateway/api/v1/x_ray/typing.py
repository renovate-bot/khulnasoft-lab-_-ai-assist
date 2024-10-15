from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator
from pydantic.types import Json

from ai_gateway.models import KindAnthropicModel, KindModelProvider

__all__ = [
    "XRayRequest",
    "XRayResponse",
]


class PackageFilePromptPayload(BaseModel):
    prompt: Annotated[str, StringConstraints(max_length=400000)]
    provider: Literal[KindModelProvider.ANTHROPIC]
    model: Literal[KindAnthropicModel.CLAUDE_2_1]


class AnyPromptComponent(BaseModel):
    type: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    payload: Json
    metadata: Optional[
        dict[
            Annotated[str, StringConstraints(max_length=100)],
            Annotated[str, StringConstraints(max_length=255)],
        ]
    ] = None

    @field_validator("metadata")
    @classmethod
    def validate_medatada(cls, dictionary):
        if dictionary is not None and len(dictionary) > 10:
            raise ValueError("metadata cannot has more than 10 elements")

        return dictionary


class PackageFilePromptComponent(AnyPromptComponent):
    type: Literal["x_ray_package_file_prompt"] = "x_ray_package_file_prompt"
    payload: PackageFilePromptPayload


class XRayRequest(BaseModel):
    prompt_components: Annotated[
        List[PackageFilePromptComponent], Field(min_length=1, max_length=1)
    ]


class XRayResponse(BaseModel):
    response: str
