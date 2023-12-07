import os
from pathlib import Path
from unittest import mock

import pytest

from ai_gateway import Config, Project

test_data = dict(
    google_vertex_ai_credentials="path/key.json",
    api_host="localhost",
    api_port=8080,
    metrics_host="127.0.0.1",
    metrics_port=8082,
    docs_url="docs",
    openapi_url="openapi",
    redoc_url="redoc",
    bypass_auth=True,
    gitlab_url="gitlab",
    gitlab_api_url="gitlab/api/v4",
    customer_portal_base_url="https://customers.gitlab.com",
    palm_text_models=["palm_model1", "palm_model2"],
    palm_text_project="palm_project",
    palm_text_location="palm_location",
    is_third_party_ai_default=True,
    limited_access_third_party_ai={
        123: Project(id=123, full_name="full_name_1"),
        456: Project(id=456, full_name="full_name_2"),
        768: Project(id=768, full_name="full_name_3"),
    },
    third_party_rollout_percentage=50,
    snowplow_enabled=True,
    snowplow_endpoint="https://blizzard.local",
)


@pytest.fixture
def mock_env_vars(tmp_path, request):
    lines = [
        ",".join([str(project.id), project.full_name])
        for project in request.param["limited_access_third_party_ai"].values()
    ]

    tmp_file_limited_access = Path(tmp_path) / "limited_access.txt"
    text = "\n".join(lines)
    tmp_file_limited_access.write_text(text)

    envs = {
        "FASTAPI_API_HOST": request.param["api_host"],
        "FASTAPI_API_PORT": str(request.param["api_port"]),
        "FASTAPI_API_METRICS_HOST": request.param["metrics_host"],
        "FASTAPI_API_METRICS_PORT": str(request.param["metrics_port"]),
        "FASTAPI_DOCS_URL": request.param["docs_url"],
        "FASTAPI_OPENAPI_URL": request.param["openapi_url"],
        "FASTAPI_REDOC_URL": request.param["redoc_url"],
        "AUTH_BYPASS_EXTERNAL": str(int(request.param["bypass_auth"])),
        "GITLAB_API_URL": request.param["gitlab_api_url"],
        "GITLAB_URL": request.param["gitlab_url"],
        "CUSTOMER_PORTAL_BASE_URL": request.param["customer_portal_base_url"],
        "PALM_TEXT_MODEL_NAME": ",".join(request.param["palm_text_models"]),
        "PALM_TEXT_PROJECT": request.param["palm_text_project"],
        "PALM_TEXT_LOCATION": request.param["palm_text_location"],
        "F_THIRD_PARTY_AI_LIMITED_ACCESS": str(tmp_file_limited_access),
        "F_IS_THIRD_PARTY_AI_DEFAULT": str(
            int(request.param["is_third_party_ai_default"])
        ),
        "F_THIRD_PARTY_ROLLOUT_PERCENTAGE": str(
            int(request.param["third_party_rollout_percentage"])
        ),
        "SNOWPLOW_ENABLED": str(int(request.param["snowplow_enabled"])),
        "SNOWPLOW_ENDPOINT": request.param["snowplow_endpoint"],
    }

    with mock.patch.dict(os.environ, envs):
        yield


@pytest.mark.parametrize("mock_env_vars", [test_data], indirect=True)
@pytest.mark.parametrize("configuration", [test_data])
def test_config(mock_env_vars, configuration):
    config = Config()

    assert config.fastapi.api_host == configuration["api_host"]
    assert config.fastapi.api_port == configuration["api_port"]
    assert config.fastapi.metrics_host == configuration["metrics_host"]
    assert config.fastapi.metrics_port == configuration["metrics_port"]

    assert config.fastapi.docs_url == configuration["docs_url"]
    assert config.fastapi.openapi_url == configuration["openapi_url"]
    assert config.fastapi.redoc_url == configuration["redoc_url"]
    assert config.fastapi.uvicorn_logger is not None

    assert config.auth.bypass == configuration["bypass_auth"]
    assert config.auth.gitlab_api_base_url == configuration["gitlab_api_url"]
    assert config.auth.gitlab_base_url == configuration["gitlab_url"]
    assert (
        config.auth.customer_portal_base_url
        == configuration["customer_portal_base_url"]
    )

    assert config.palm_text_model.names == configuration["palm_text_models"]
    assert config.palm_text_model.project == configuration["palm_text_project"]
    assert config.palm_text_model.location == configuration["palm_text_location"]

    assert (
        config.feature_flags.limited_access_third_party_ai
        == configuration["limited_access_third_party_ai"]
    )
    assert (
        config.feature_flags.is_third_party_ai_default
        == configuration["is_third_party_ai_default"]
    )
    assert (
        config.feature_flags.third_party_rollout_percentage
        == configuration["third_party_rollout_percentage"]
    )

    assert config.tracking.snowplow_enabled == configuration["snowplow_enabled"]
    assert config.tracking.snowplow_endpoint == configuration["snowplow_endpoint"]


@pytest.mark.parametrize(
    "use_fake_models,expected",
    [
        ("false", "real"),
        ("true", "fake"),
    ],
)
def test_config_fake_models(use_fake_models, expected):
    with mock.patch.dict(os.environ, {"USE_FAKE_MODELS": use_fake_models}):
        config = Config()

        assert config.palm_text_model.real_or_fake == expected


@pytest.mark.parametrize(
    ("f_value", "expected"),
    [
        ("", []),
        ("func1", ["func1"]),
        ("func1;func2", ["func1", "func2"]),
        ("func1;func2;func3", ["func1", "func2", "func3"]),
    ],
)
def test_config_code_suggestions_excl_post_proc(f_value: str, expected: list):
    with mock.patch.dict(os.environ, {"F_CODE_SUGGESTIONS_EXCL_POST_PROC": f_value}):
        config = Config()

        assert config.feature_flags.code_suggestions_excl_post_proc == expected
