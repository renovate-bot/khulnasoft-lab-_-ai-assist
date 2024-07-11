from unittest.mock import patch

import pytest

from ai_gateway.searches.container import _init_vertex_search_service_client


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {"mock_model_responses": False, "custom_models_enabled": False},
            True,
        ),
        (
            {"mock_model_responses": False, "custom_models_enabled": True},
            False,
        ),
        (
            {"mock_model_responses": True, "custom_models_enabled": False},
            False,
        ),
    ],
)
def test_init_vertex_search_service_client(args, expected_init):
    with patch(
        "google.cloud.discoveryengine.SearchServiceAsyncClient"
    ) as mock_search_client:
        next(_init_vertex_search_service_client(**args))

        if expected_init:
            mock_search_client.assert_called_once()
        else:
            mock_search_client.assert_not_called()
