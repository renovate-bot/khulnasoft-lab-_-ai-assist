import os
from unittest import mock

import pytest

from ai_gateway.config import (
    Config,
    ConfigAuth,
    ConfigCustomModels,
    ConfigDefaultPrompts,
    ConfigFastApi,
    ConfigGoogleCloudPlatform,
    ConfigGoogleCloudProfiler,
    ConfigInstrumentator,
    ConfigLogging,
    ConfigModelConcurrency,
    ConfigSnowplow,
    ConfigVertexSearch,
    ConfigVertexTextModel,
    FFlagsCodeSuggestions,
)

# pylint: disable=direct-environment-variable-reference


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
                mock_model_responses=True,
            ),
        ),
    ],
)
def test_config_base(values: dict, expected: Config):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None, _env_prefix="AIGW_")  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.fastapi == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuth()),
        ({"AIGW_AUTH__BYPASS_EXTERNAL": "yes"}, ConfigAuth(bypass_external=True)),
    ],
)
def test_config_auth_bypass_external(values: dict, expected: ConfigAuth):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.auth == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigAuth()),
        (
            {"AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER": "yes"},
            ConfigAuth(bypass_external_with_header=True),
        ),
    ],
)
def test_config_auth_bypass_external_with_header(values: dict, expected: ConfigAuth):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

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
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.instrumentator == expected


@pytest.mark.parametrize(
    (
        "values",
        "expected_google_cloud_platform",
        "expected_vertex_text_model",
        "expected_vertex_search",
    ),
    [
        (
            {},
            ConfigGoogleCloudPlatform(),
            ConfigVertexTextModel(),
            ConfigVertexSearch(),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_GOOGLE_CLOUD_PLATFORM__SERVICE_ACCOUNT_JSON_KEY": "global-secret",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
                service_account_json_key="global-secret",
            ),
            ConfigVertexTextModel(
                project="global-project",
                service_account_json_key="global-secret",
            ),
            ConfigVertexSearch(
                project="global-project",
                service_account_json_key="global-secret",
            ),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "specific-project-1",
                "AIGW_VERTEX_SEARCH__PROJECT": "specific-project-2",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
            ),
            ConfigVertexTextModel(
                project="specific-project-1",
            ),
            ConfigVertexSearch(
                project="specific-project-2",
            ),
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "global-project",
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "",
                "AIGW_VERTEX_SEARCH__PROJECT": "",
            },
            ConfigGoogleCloudPlatform(
                project="global-project",
            ),
            ConfigVertexTextModel(
                project="",
            ),
            ConfigVertexSearch(
                project="",
            ),
        ),
    ],
)
def test_config_google_cloud_platform(
    values: dict,
    expected_google_cloud_platform: ConfigGoogleCloudPlatform,
    expected_vertex_text_model: ConfigVertexTextModel,
    expected_vertex_search: ConfigVertexSearch,
):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.google_cloud_platform == expected_google_cloud_platform
        assert config.vertex_text_model == expected_vertex_text_model
        assert config.vertex_search == expected_vertex_search


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigCustomModels(enabled=False)),
        (
            {
                "AIGW_CUSTOM_MODELS__ENABLED": "True",
            },
            ConfigCustomModels(
                enabled=True,
            ),
        ),
    ],
)
def test_custom_models(values: dict, expected: ConfigCustomModels):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)

        assert config.custom_models == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigVertexTextModel()),
        (
            {
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "project",
                "AIGW_VERTEX_TEXT_MODEL__LOCATION": "location",
                "AIGW_VERTEX_TEXT_MODEL__ENDPOINT": "endpoint",
                "RUNWAY_REGION": "test-case1",  # ignored
            },
            ConfigVertexTextModel(
                project="project",
                location="location",
                endpoint="endpoint",
            ),
        ),
        (
            {
                "AIGW_VERTEX_TEXT_MODEL__PROJECT": "project",
                "RUNWAY_REGION": "test-case1",
            },
            ConfigVertexTextModel(
                project="project",
                location="test-case1",
                endpoint="test-case1-aiplatform.googleapis.com",
            ),
        ),
    ],
)
def test_config_vertex_text_model(values: dict, expected: ConfigVertexTextModel):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.vertex_text_model == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigVertexSearch()),
        (
            {
                "AIGW_VERTEX_SEARCH__PROJECT": "project",
            },
            ConfigVertexSearch(
                project="project",
                fallback_datastore_version="",
            ),
        ),
        (
            {
                "AIGW_VERTEX_SEARCH__PROJECT": "project",
                "AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION": "17.0",
            },
            ConfigVertexSearch(
                project="project",
                fallback_datastore_version="17.0",
            ),
        ),
    ],
)
def test_config_vertex_search(values: dict, expected: ConfigVertexSearch):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.vertex_search == expected


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
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.model_engine_concurrency_limits == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, ConfigDefaultPrompts()),
        (
            {"AIGW_DEFAULT_PROMPTS": '{"chat/react": "vertex"}'},
            ConfigDefaultPrompts({"chat/react": "vertex"}),
        ),
    ],
)
def test_config_default_prompts(values: dict, expected: ConfigDefaultPrompts):
    with mock.patch.dict(os.environ, values, clear=True):
        config = Config(_env_file=None)  # type: ignore[call-arg]

        assert config.default_prompts == expected


# pylint: enable=direct-environment-variable-reference
