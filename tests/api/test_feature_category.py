from unittest import mock
from unittest.mock import patch

import pytest

from ai_gateway.api.feature_category import feature_category


@pytest.mark.asyncio
@mock.patch("ai_gateway.api.feature_category._FEATURE_CATEGORIES", ["awesome_category"])
async def test_feature_category():
    @feature_category("awesome_category")
    async def to_be_decorated():
        pass

    with patch("ai_gateway.api.feature_category.context") as mock_context:
        await to_be_decorated()

        mock_context.__setitem__.assert_called_once_with(
            "meta.feature_category", "awesome_category"
        )


def test_unknown_feature_category():
    with pytest.raises(ValueError) as error:
        feature_category("not_exist")

    assert str(error.value) == "Invalid feature category: not_exist"
