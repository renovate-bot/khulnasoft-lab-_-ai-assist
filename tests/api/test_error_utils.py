from typing import Optional

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from ai_gateway.api.error_utils import capture_validation_errors


class SampleModel(BaseModel):
    sample_required_str_field: str
    sample_required_bool_field: bool
    sample_required_int_field: int
    sample_optional_str_field: Optional[str] = None


@capture_validation_errors()
async def sample_function(data: dict):
    model = SampleModel(**data)
    return model


@pytest.mark.asyncio
async def test_valid_data():
    valid_data = {
        "sample_required_str_field": "a string",
        "sample_required_bool_field": True,
        "sample_required_int_field": 100,
    }

    result = await sample_function(valid_data)

    assert isinstance(result, SampleModel)
    assert result.sample_required_str_field == "a string"
    assert result.sample_required_bool_field is True
    assert result.sample_required_int_field == 100


@pytest.mark.asyncio
async def test_invalid_data():
    invalid_data = {
        "sample_required_str_field": "a string",
        "sample_required_bool_field": 100,
        "sample_required_int_field": 100,
        "sample_optional_str_field": "a string",
    }

    with pytest.raises(HTTPException) as ex:
        await sample_function(invalid_data)

    assert ex.value.status_code == 422
    assert "sample_required_bool_field" in ex.value.detail
    assert (
        "Input should be a valid boolean, unable to interpret input" in ex.value.detail
    )
