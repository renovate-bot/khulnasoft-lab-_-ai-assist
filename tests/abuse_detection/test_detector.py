from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from structlog.testing import capture_logs

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.models.anthropic import AnthropicChatModel
from ai_gateway.models.base_text import TextGenModelOutput


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.url = "http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages"
    request.headers = {
        "X-Gitlab-Instance-Id": "test_instance_id",
        "X-Gitlab-Global-User-Id": "test_global_user_id",
        "x-gitlab-unit-primitive": "test_unit_primitive",
    }
    return request


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AnthropicChatModel)
    model.generate = AsyncMock(return_value=TextGenModelOutput(text="0.2"))
    return model


@pytest.fixture
def abuse_detector(mock_model):
    return AbuseDetector(enabled=True, sampling_rate=1.0, model=mock_model)


@pytest.mark.asyncio
async def test_should_detect_enabled(abuse_detector):
    assert abuse_detector.should_detect() is True


@pytest.mark.asyncio
async def test_should_detect_disabled(mock_model):
    detector = AbuseDetector(enabled=False, sampling_rate=1.0, model=mock_model)
    assert detector.should_detect() is False


@pytest.mark.asyncio
async def test_eval(mock_request, abuse_detector):
    body = '{"messages": [{"role": "user", "content": "How can I create an issue in GitLab?"}]}'
    expected_usecase = "Asking a question about how to use GitLab."

    score = await abuse_detector._eval(mock_request, body, expected_usecase)
    assert float(score) == 0.2


@pytest.mark.asyncio
async def test_eval_failure(mock_request):
    body = '{"messages": [{"role": "user", "content": "How can I create an issue in GitLab?"}]}'
    expected_usecase = "Asking a question about how to use GitLab."

    model = MagicMock(spec=AnthropicChatModel)
    model.generate = AsyncMock(return_value=TextGenModelOutput(text="invalid text"))
    abuse_detector = AbuseDetector(enabled=True, sampling_rate=1.0, model=model)

    with capture_logs() as cap_logs:
        result = await abuse_detector._eval(mock_request, body, expected_usecase)

    assert result is None
    assert len(cap_logs) == 2
    assert cap_logs[1]["exception_class"] == "ValueError"


@pytest.mark.asyncio
async def test_detect(mock_request, abuse_detector):
    body = '{"messages": [{"role": "user", "content": "How can I create an issue in GitLab?"}]}'
    expected_usecase = "Asking a question about how to use GitLab."

    await abuse_detector.detect(mock_request, body, expected_usecase)
    abuse_detector.model.generate.assert_called_once()


@pytest.mark.asyncio
async def test_report(mock_request, abuse_detector):
    score = 0.2

    with capture_logs() as cap_logs:
        with patch("prometheus_client.Histogram.labels") as mock_histograms:
            abuse_detector._report(score, mock_request)

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == "abuse request score"
    assert cap_logs[0]["score"] == score
    assert cap_logs[0]["instance_id"] == "test_instance_id"
    assert cap_logs[0]["global_user_id"] == "test_global_user_id"
    assert cap_logs[0]["unit_primitive"] == "test_unit_primitive"

    assert mock_histograms.mock_calls == [
        mock.call(
            instance_id="test_instance_id",
            global_user_id="test_global_user_id",
            unit_primitive="test_unit_primitive",
        ),
        mock.call().observe(score),
    ]
