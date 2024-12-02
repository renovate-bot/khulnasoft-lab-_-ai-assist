from unittest.mock import Mock

import pytest
from fastapi import Request

from ai_gateway.api.snowplow_context import (
    X_GITLAB_REALM_HEADER,
    get_snowplow_code_suggestion_context,
)


@pytest.fixture
def mock_request():
    request = Mock(spec=Request)
    request.user = None
    return request


@pytest.mark.parametrize(
    "headers, expected_realm",
    [
        ({X_GITLAB_REALM_HEADER: "GitLab.com"}, "gitlab.com"),
        ({X_GITLAB_REALM_HEADER: ""}, ""),
        ({}, ""),
    ],
    ids=[
        "has realm header",
        "blank realm header",
        "no realm header",
    ],
)
def test_get_snowplow_code_suggestion_context_realm_header(
    mock_request, headers, expected_realm
):
    mock_request.headers = headers

    context = get_snowplow_code_suggestion_context(req=mock_request, region="us-east-1")

    assert context.gitlab_realm == expected_realm
