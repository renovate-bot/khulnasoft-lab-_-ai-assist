from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, StringConstraints

__all__ = [
    "ApplicationRequest",
    "EventRequest",
]


class ApplicationRequest(BaseModel):
    client_id: Annotated[str, StringConstraints(max_length=255)]
    client_secret: Annotated[str, StringConstraints(max_length=1000)]
    instance_url: Annotated[str, StringConstraints(max_length=500)]
    redirect_url: Annotated[str, StringConstraints(max_length=500)]
    role_arn: Annotated[str, StringConstraints(max_length=2048)]


class EventRequestPayload(BaseModel):
    command: Annotated[str, StringConstraints(max_length=255)]
    source: Annotated[str, StringConstraints(max_length=255)]
    project_path: Annotated[str, StringConstraints(max_length=1024)]
    project_id: Annotated[str, StringConstraints(max_length=1024)]
    note_id: Optional[Annotated[str, StringConstraints(max_length=1024)]]
    discussion_id: Annotated[str, StringConstraints(max_length=1024)]


class EventMergeRequestPayload(EventRequestPayload):
    source: Literal["merge_request"]
    merge_request_id: Annotated[str, StringConstraints(max_length=1024)]
    merge_request_iid: Annotated[str, StringConstraints(max_length=1024)]
    source_branch: Annotated[str, StringConstraints(max_length=1024)]
    target_branch: Annotated[str, StringConstraints(max_length=1024)]
    last_commit_id: Annotated[str, StringConstraints(max_length=1024)]
    start_sha: Optional[Annotated[str, StringConstraints(max_length=1024)]] = None
    head_sha: Optional[Annotated[str, StringConstraints(max_length=1024)]] = None
    file_path: Optional[Annotated[str, StringConstraints(max_length=1024)]] = None
    comment_start_line: Optional[Annotated[str, StringConstraints(max_length=1024)]] = (
        None
    )
    comment_end_line: Optional[Annotated[str, StringConstraints(max_length=1024)]] = (
        None
    )
    user_message: Optional[Annotated[str, StringConstraints(max_length=1024)]] = None


class EventIssuePayload(EventRequestPayload):
    source: Literal["issue"]
    issue_id: Annotated[str, StringConstraints(max_length=1024)]
    issue_iid: Annotated[str, StringConstraints(max_length=1024)]


class EventRequest(BaseModel):
    role_arn: Annotated[str, StringConstraints(max_length=2048)]
    code: Annotated[str, StringConstraints(max_length=255)]
    payload: Union[EventMergeRequestPayload, EventIssuePayload] = Field(
        discriminator="source"
    )
