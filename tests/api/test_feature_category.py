# from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from enum import Enum
from unittest import mock
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from starlette_context import context, request_cycle_context

from ai_gateway.api.feature_category import (
    feature_categories,
    feature_category,
    get_feature_category,
)


class DummyGitLabFeatureCategory(str, Enum):
    AWESOME_CATEGORY_1 = "awesome_category_1"
    AWESOME_CATEGORY_2 = "awesome_category_2"


class DummyGitLabUnitPrimitive(str, Enum):
    AWESOME_FEATURE_1 = "awesome_feature_1"
    AWESOME_FEATURE_2 = "awesome_feature_2"


@pytest.fixture
def patch_feature_category():
    patcher = patch("ai_gateway.api.feature_category.GitLabFeatureCategory")
    mock_thing = patcher.start()
    mock_thing.side_effect = DummyGitLabFeatureCategory
    yield
    patcher.stop()


@pytest.fixture
def patch_unit_primitive():
    patcher = patch("ai_gateway.api.feature_category.GitLabUnitPrimitive")
    mock_thing = patcher.start()
    mock_thing.side_effect = DummyGitLabUnitPrimitive
    yield
    patcher.stop()


@pytest.mark.asyncio
async def test_feature_category(patch_feature_category):
    @feature_category(DummyGitLabFeatureCategory.AWESOME_CATEGORY_1)
    async def to_be_decorated():
        pass

    with patch("ai_gateway.api.feature_category.context") as mock_context:
        await to_be_decorated()

        mock_context.__setitem__.assert_called_once_with(
            "meta.feature_category", DummyGitLabFeatureCategory.AWESOME_CATEGORY_1
        )


def test_unknown_feature_category():
    with pytest.raises(ValueError) as error:
        feature_category("not_exist")

    assert str(error.value) == "Invalid feature category: not_exist"


def test_get_feature_category():
    with request_cycle_context({"meta.feature_category": "awesome_category"}):
        assert get_feature_category() == "awesome_category"

    # When there value isn't set in the context
    with request_cycle_context({}):
        assert get_feature_category() == "unknown"

    # When the context isn't initialized
    assert get_feature_category() == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected_unit_primitive", "expected_category", "expected_error"),
    [
        (
            {"x-gitlab-unit-primitive": DummyGitLabUnitPrimitive.AWESOME_FEATURE_1},
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_1,
            DummyGitLabFeatureCategory.AWESOME_CATEGORY_1,
            "",
        ),
        (
            {"x-gitlab-unit-primitive": DummyGitLabUnitPrimitive.AWESOME_FEATURE_2},
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_2,
            DummyGitLabFeatureCategory.AWESOME_CATEGORY_2,
            "",
        ),
        ({}, None, None, "400: Missing x-gitlab-unit-primitive header"),
        (
            {"x-gitlab-unit-primitive": "unknown"},
            None,
            None,
            "400: This endpoint cannot be used for unknown purpose",
        ),
    ],
)
async def test_feature_categories(
    headers,
    expected_error,
    expected_unit_primitive,
    expected_category,
    patch_unit_primitive,
    patch_feature_category,
):
    @feature_categories(
        {
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_1: DummyGitLabFeatureCategory.AWESOME_CATEGORY_1,
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_2: DummyGitLabFeatureCategory.AWESOME_CATEGORY_2,
        }
    )
    async def to_be_decorated(request: Request):
        pass

    request = Mock(spec=Request)
    request.headers = headers

    if expected_error:
        with pytest.raises(HTTPException, match=expected_error):
            await to_be_decorated(request=request)
    else:
        with patch("ai_gateway.api.feature_category.context") as mock_context:
            await to_be_decorated(request=request)

            mock_context.__setitem__.assert_has_calls(
                [
                    mock.call(
                        "meta.feature_category",
                        expected_category,
                    ),
                    mock.call(
                        "meta.unit_primitive",
                        expected_unit_primitive,
                    ),
                ]
            )
