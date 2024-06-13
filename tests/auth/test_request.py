from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.auth.request import authorize_with_unit_primitive_header
from ai_gateway.auth.user import GitLabUser


@pytest.fixture
def mock_request():
    request = Mock(spec=Request)
    request.headers = {}
    request.user = Mock(spec=GitLabUser)
    return request


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_missing_header(mock_request):
    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == f"Missing {X_GITLAB_UNIT_PRIMITIVE} header"


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_unauthorized(mock_request):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = "awesome_feature"
    mock_request.user.can.return_value = False

    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Unauthorized to access awesome_feature"


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_authorized(mock_request):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = "awesome_feature"
    mock_request.user.can.return_value = True

    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    result = await dummy_func(mock_request)
    assert result == "Success"
