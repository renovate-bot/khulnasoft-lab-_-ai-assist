from typing import Annotated

from pydantic import BaseModel, StringConstraints

__all__ = [
    "ApplicationRequest",
]


class ApplicationRequest(BaseModel):
    client_id: Annotated[str, StringConstraints(max_length=255)]
    client_secret: Annotated[str, StringConstraints(max_length=1000)]
    instance_url: Annotated[str, StringConstraints(max_length=500)]
    redirect_url: Annotated[str, StringConstraints(max_length=500)]
    role_arn: Annotated[str, StringConstraints(max_length=2048)]
