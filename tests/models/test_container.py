import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_gateway.models.container import _init_vertex_grpc_client


@pytest.mark.parametrize(
    ("args", "expected_init", "expected_key_to_file"),
    [
        (
            {"endpoint": "test", "json_key": "", "mock_model_responses": False},
            True,
            False,
        ),
        (
            {
                "endpoint": "test",
                "json_key": '{ "type": "service_account" }',
                "mock_model_responses": False,
            },
            True,
            True,
        ),
        (
            {"endpoint": "test", "json_key": "", "mock_model_responses": True},
            False,
            False,
        ),
    ],
)
def test_init_vertex_grpc_client(args, expected_init, expected_key_to_file):
    with patch(
        # "google.cloud.aiplatform.gapic.PredictionServiceAsyncClient"
        "ai_gateway.models.container.grpc_connect_vertex"
    ) as mock_grpc_client:
        next(_init_vertex_grpc_client(**args))

        if expected_init:
            mock_grpc_client.assert_called_once_with({"api_endpoint": args["endpoint"]})
        else:
            mock_grpc_client.assert_not_called()

        if expected_key_to_file:
            with open("/tmp/vertex-client.json", "r") as f:
                assert f.read() == args["json_key"]
            assert (
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                == "/tmp/vertex-client.json"
            )
