import os
from unittest import mock

import pytest

from ai_gateway.config import (
    Config,
    ConfigAuth,
    ConfigFastApi,
    ConfigGoogleCloudProfiler,
    ConfigInstrumentator,
    ConfigLogging,
    ConfigModelConcurrency,
    ConfigSnowplow,
    ConfigVertexTextModel,
    FFlagsCodeSuggestions,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "AIGW_GITLAB_URL": "http://gitlab.test",
                "AIGW_GITLAB_API_URL": "http://api.gitlab.test",
                "AIGW_CUSTOMER_PORTAL_URL": "http://customer.gitlab.test",
                "AIGW_MOCK_MODEL_RESPONSES": "true",
            },
            Config(
                gitlab_url="http://gitlab.test",
                gitlab_api_url="http://api.gitlab.test",
                customer_portal_url="http://customer.gitlab.test",
                # pydantic-settings does not allow omitting the prefix if validation_alias is set for the field
                aigw_mock_model_responses=True,
            ),
        ),
    ],
)
def test_config_base(values: dict, expected: Config):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None, _env_prefix="AIGW_")

        keys = {
            "gitlab_url",
            "gitlab_api_url",
            "customer_portal_url",
            "mock_model_responses",
        }

        actual = config.model_dump(include=keys)
        assert actual == expected.model_dump(include=keys)
        assert len(actual) == len(keys)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # pydantic-settings does not allow omitting the prefix if validation_alias is set for the field
        ({"AIGW_MOCK_MODEL_RESPONSES": "true"}, Config(aigw_mock_model_responses=True)),
        (
            {"AIGW_USE_FAKE_MODELS": "true"},
            Config(aigw_mock_model_responses=True),
        ),
    ],
)
def test_mock_model_responses_b_compatibility(values: dict, expected: Config):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.mock_model_responses == expected.mock_model_responses


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigLogging()),
        (
            {
                "AIGW_LOGGING__LEVEL": "DEBUG",
                "AIGW_LOGGING__FORMAT_JSON": "no",
                "AIGW_LOGGING__TO_FILE": "/file/file1.text",
            },
            ConfigLogging(level="DEBUG", format_json=False, to_file="/file/file1.text"),
        ),
    ],
)
def test_config_logging(values: dict, expected: ConfigLogging):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.logging == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigFastApi()),
        (
            {
                "AIGW_FASTAPI__API_HOST": "localhost",
                "AIGW_FASTAPI__API_PORT": "80",
                "AIGW_FASTAPI__METRICS_HOST": "localhost",
                "AIGW_FASTAPI__METRICS_PORT": "82",
                "AIGW_FASTAPI__UVICORN_LOGGER": '{"key": "value"}',
                "AIGW_FASTAPI__DOCS_URL": "docs.test",
                "AIGW_FASTAPI__OPENAPI_URL": "openapi.test",
                "AIGW_FASTAPI__REDOC_URL": "redoc.test",
                "AIGW_FASTAPI__RELOAD": "True",
            },
            ConfigFastApi(
                api_host="localhost",
                api_port=80,
                metrics_host="localhost",
                metrics_port=82,
                uvicorn_logger={"key": "value"},
                docs_url="docs.test",
                openapi_url="openapi.test",
                redoc_url="redoc.test",
                reload=True,
            ),
        ),
    ],
)
def test_config_fastapi(values: dict, expected: ConfigFastApi):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.fastapi == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuth()),
        ({"AIGW_AUTH__BYPASS_EXTERNAL": "yes"}, ConfigAuth(bypass_external=True)),
    ],
)
def test_config_auth(values: dict, expected: ConfigAuth):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.auth == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigGoogleCloudProfiler()),
        (
            {
                "AIGW_GOOGLE_CLOUD_PROFILER__ENABLED": "yes",
                "AIGW_GOOGLE_CLOUD_PROFILER__VERBOSE": "1",
                "AIGW_GOOGLE_CLOUD_PROFILER__PERIOD_MS": "5",
            },
            ConfigGoogleCloudProfiler(enabled=True, verbose=1, period_ms=5),
        ),
    ],
)
def test_config_google_cloud_profiler(
    values: dict, expected: ConfigGoogleCloudProfiler
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.google_cloud_profiler == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, FFlagsCodeSuggestions()),
        (
            {"AIGW_F__CODE_SUGGESTIONS__EXCL_POST_PROC": '["func1", "func2"]'},
            FFlagsCodeSuggestions(excl_post_proc=["func1", "func2"]),
        ),
    ],
)
def test_config_f_flags_code_suggestions(values: dict, expected: FFlagsCodeSuggestions):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.f.code_suggestions == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigSnowplow()),
        (
            {
                "AIGW_SNOWPLOW__ENABLED": "yes",
                "AIGW_SNOWPLOW__ENDPOINT": "endpoint.test",
                "AIGW_SNOWPLOW__BATCH_SIZE": "8",
                "AIGW_SNOWPLOW__THREAD_COUNT": "7",
            },
            ConfigSnowplow(
                enabled=True, endpoint="endpoint.test", thread_count=7, batch_size=8
            ),
        ),
    ],
)
def test_config_snowplow(values: dict, expected: ConfigSnowplow):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.snowplow == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigInstrumentator()),
        (
            {
                "AIGW_INSTRUMENTATOR__THREAD_MONITORING_ENABLED": "True",
                "AIGW_INSTRUMENTATOR__THREAD_MONITORING_INTERVAL": "45",
            },
            ConfigInstrumentator(
                thread_monitoring_enabled=True, thread_monitoring_interval=45
            ),
        ),
    ],
)
def test_config_instrumentator(values: dict, expected: ConfigInstrumentator):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.instrumentator == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigVertexTextModel()),
        (
            {
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "project",
                "AIGW_VERTEX_TEXT_MODEL__LOCATION": "location",
                "AIGW_VERTEX_TEXT_MODEL__ENDPOINT": "endpoint",
                "AIGW_VERTEX_TEXT_MODEL__JSON_KEY": "secret",
            },
            ConfigVertexTextModel(
                project="project",
                location="location",
                endpoint="endpoint",
                json_key="secret",
            ),
        ),
    ],
)
def test_config_vertex_text_model(values: dict, expected: ConfigVertexTextModel):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.vertex_text_model == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigModelConcurrency()),
        (
            {"AIGW_MODEL_ENGINE_CONCURRENCY_LIMITS": '{"engine": {"model": 10}}'},
            ConfigModelConcurrency({"engine": {"model": 10}}),
        ),
    ],
)
def test_config_model_concurrency(values: dict, expected: ConfigModelConcurrency):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.model_engine_concurrency_limits == expected
