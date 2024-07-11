import random
import textwrap

import structlog
from fastapi import Request
from prometheus_client import Histogram
from pydantic import BaseModel

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
)
from ai_gateway.models.anthropic import AnthropicChatModel
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.tracking import log_exception

log = structlog.stdlib.get_logger("abuse_detection")

METRIC_LABELS = ["instance_id", "global_user_id", "unit_primitive"]

ABUSE_SCORE = Histogram(
    "abuse_request_scores",
    "Score of the abuse requests",
    METRIC_LABELS,
    buckets=(0.0, 0.3, 0.5, 0.7, 1.0),
)


__all__ = ["AbuseDetector"]


class ModelRequest(BaseModel):
    system: str
    messages: list[Message]
    stop_sequences: list[str]
    max_tokens: int


class AbuseDetector:
    SYSTEM_PROMPT = textwrap.dedent(
        """
    You're an abuse detector that evaluates if a user request is aligned with the expected use case.
    """
    ).strip()

    SCORE_TAG_START = "<score>"
    SCORE_TAG_END = "</score>"

    # https://docs.anthropic.com/en/docs/use-examples
    QUERY_TEMPLATE = textwrap.dedent(
        """
    <user_request_url>{user_request_url}</user_request_url>
    <user_request_body>{user_request_body}</user_request_body>
    <expected_usecase>{expected_usecase}</expected_usecase>
    """
    ).strip()

    EXAMPLE_TEMPLATE = QUERY_TEMPLATE + "\n<score>{score}</score>"
    EXAMPLES = [
        EXAMPLE_TEMPLATE.format(
            user_request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
            user_request_body='{"messages": [{ "role": "user", "content": "How can I create an issue in GitLab?" }]}',
            expected_usecase="Asking a question about how to use GitLab.",
            score=0.1,
        ),
        EXAMPLE_TEMPLATE.format(
            user_request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
            user_request_body='{"messages": [{ "role": "user", "content": "How can I fork a repository in GitLab?" }]}',
            expected_usecase="Asking about how to resolve a security vulnerability in the given CVE or code.",
            score=1.0,
        ),
    ]
    EXAMPLE_PROMPT = "".join([f"\n<example>\n{e}\n</example>\n" for e in EXAMPLES])

    BASE_USER_PROMPT = (
        textwrap.dedent(
            """
    Evaluate if the user request is aligned with the expected use case.
    Give a score between 0.0-1.0 (higher is more misaligned) for the criteria.

    The evaluation score should be placed in the XML tag <score></score>.
    """
        ).strip()
        + EXAMPLE_PROMPT
    )

    def __init__(self, enabled: bool, sampling_rate: float, model: AnthropicChatModel):
        self.enabled = enabled
        self.sampling_rate = sampling_rate
        self.model = model

    def should_detect(self) -> bool:
        if not self.enabled:
            return False

        return random.random() < self.sampling_rate

    async def detect(
        self,
        request: Request,
        body: str,
        expected_usecase: str,
    ) -> None:
        result = await self._eval(request, body, expected_usecase)

        if result is not None:
            self._report(result, request)

    async def _eval(
        self,
        request: Request,
        body: str,
        expected_usecase: str,
    ) -> float | None:
        prompt = (
            self.BASE_USER_PROMPT
            + "\n"
            + self.QUERY_TEMPLATE.format(
                user_request_url=request.url,
                user_request_body=body,
                expected_usecase=expected_usecase,
            )
        )

        model_request = ModelRequest(
            system=self.SYSTEM_PROMPT,
            messages=[
                Message(role=Role.USER, content=prompt),
                # Prefilling JSON format https://docs.anthropic.com/en/docs/control-output-format
                Message(role=Role.ASSISTANT, content=AbuseDetector.SCORE_TAG_START),
            ],
            stop_sequences=[AbuseDetector.SCORE_TAG_END],
            max_tokens=8,
        )

        log.debug("abuse detector call:", **model_request.model_dump())
        response = await self.model.generate(**dict(model_request))
        assert isinstance(response, TextGenModelOutput)

        score = None

        try:
            score = float(response.text)
        except Exception as e:
            log_exception(e)

        return score

    def _report(self, score: float, request: Request):
        instance_id = request.headers.get(X_GITLAB_INSTANCE_ID_HEADER, "unknown")
        global_user_id = request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER, "unknown")
        unit_primitive = request.headers.get(X_GITLAB_UNIT_PRIMITIVE, "unknown")

        detail_labels = {
            "instance_id": instance_id,
            "global_user_id": global_user_id,
            "unit_primitive": unit_primitive,
        }

        log.info("abuse request score", score=score, **detail_labels)
        ABUSE_SCORE.labels(**detail_labels).observe(score)
