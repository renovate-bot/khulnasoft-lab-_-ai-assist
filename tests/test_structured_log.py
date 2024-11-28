from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_gateway.structured_logging import sanitize_logs


@pytest.fixture
def inputs_with_model_metadata():
    inputs = MagicMock(
        model_metadata=MagicMock(
            api_key="secret-key-456", endpoint="https://example.com"
        ),
        other_fied="other_value",
    )

    return inputs


class TestSanitizeLogs:
    def test_sanitize_api_key(self):
        # Test when api_key is present
        event_dict = {"api_key": "secret-key-123"}
        result = sanitize_logs(None, None, event_dict)
        assert result["api_key"] == "**********"

    def test_sanitize_missing_api_key(self):
        # Test when api_key is not present
        event_dict = {"other_field": "value"}
        result = sanitize_logs(None, None, event_dict)
        assert result["api_key"] is None

    def test_sanitize_inputs_with_model_metadata(self, inputs_with_model_metadata):
        event_dict = {"inputs": inputs_with_model_metadata}

        result = sanitize_logs(None, None, event_dict)

        assert result["inputs"].model_metadata.api_key == "**********"
        assert result["inputs"].model_metadata.endpoint == "https://example.com"
        assert result["inputs"].other_fied == "other_value"

    def test_sanitize_inputs_without_model_metadata(self):
        # Test when inputs exist but without model_metadata
        inputs = SimpleNamespace(other_field="test")
        event_dict = {"inputs": inputs}

        result = sanitize_logs(None, None, event_dict)
        assert result["inputs"].other_field == "test"

    def test_sanitize_no_inputs(self):
        # Test when no inputs field exists
        event_dict = {"some_field": "value"}
        result = sanitize_logs(None, None, event_dict)
        assert "inputs" not in result
        assert result["some_field"] == "value"
