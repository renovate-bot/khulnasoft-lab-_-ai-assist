import pytest
from google.api_core.exceptions import (
    DuplicateCredentialArgs,
    GoogleAPICallError,
    GoogleAPIError,
    InternalServerError,
    PermissionDenied,
    RetryError,
)
from google.rpc.error_details_pb2 import ErrorInfo

from ai_gateway.models import VertexAPIConnectionError, VertexAPIStatusError


class TestVertexAPIConnectionError:
    @pytest.mark.parametrize(
        ("original_error", "expected_error_string"),
        [
            (
                RetryError(message="retrying", cause=RuntimeError),
                "Vertex Model API error: RetryError retrying",
            ),
            (
                DuplicateCredentialArgs(),
                "Vertex Model API error: DuplicateCredentialArgs",
            ),
        ],
    )
    def test_from_exception(
        self, original_error: GoogleAPIError, expected_error_string: str
    ):
        wrapped_error = VertexAPIConnectionError.from_exception(original_error)

        assert str(wrapped_error) == expected_error_string


class TestVertexAPIStatusError:
    @pytest.mark.parametrize(
        ("original_error", "expected_error_string"),
        [
            (
                PermissionDenied(
                    message="Permission denied on resource project abc",
                    details=[
                        ErrorInfo(
                            reason="CONSUMER_INVALID",
                            metadata={
                                "consumer": "projects/uknown-project-id",
                                "service": "aiplatform.googleapis.com",
                            },
                        )
                    ],
                ),
                "403 Vertex Model API error: PermissionDenied Permission denied on resource project abc "
                '[reason: "CONSUMER_INVALID"\n'
                'metadata {\n  key: "service"\n  value: "aiplatform.googleapis.com"\n}\n'
                'metadata {\n  key: "consumer"\n  value: "projects/uknown-project-id"\n}\n]',
            ),
            (
                InternalServerError(message="Something went wrong"),
                "500 Vertex Model API error: InternalServerError Something went wrong",
            ),
        ],
    )
    def test_from_exception(
        self, original_error: GoogleAPICallError, expected_error_string: str
    ):
        wrapped_error = VertexAPIStatusError.from_exception(original_error)

        assert str(wrapped_error) == expected_error_string
