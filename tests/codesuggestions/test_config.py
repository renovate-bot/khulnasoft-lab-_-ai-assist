import os
from pathlib import Path

import pytest
from unittest import mock

from codesuggestions import Config, Project

test_data = dict(
    google_vertex_ai_credentials="path/key.json",

    triton_host="localhost",
    triton_port=5000,

    api_host="localhost",
    api_port=8080,
    metrics_host="127.0.0.1",
    metrics_port=8082,
    docs_url="docs",
    openapi_url="openapi",
    redoc_url="redoc",

    bypass_auth=True,
    gitlab_url="gitlab",

    palm_text_model="palm_model",
    palm_text_project="palm_project",
    palm_text_location="palm_location",

    is_third_party_ai_default=True,
    limited_access_third_party_ai={
        123: Project(id=123, full_name="full_name_1"),
        456: Project(id=456, full_name="full_name_2"),
        768: Project(id=768, full_name="full_name_3"),
    }
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
        "TRITON_HOST": request.param["triton_host"],
        "TRITON_PORT": str(request.param["triton_port"]),

        "FASTAPI_API_HOST": request.param["api_host"],
        "FASTAPI_API_PORT": str(request.param["api_port"]),
        "FASTAPI_API_METRICS_HOST": request.param["metrics_host"],
        "FASTAPI_API_METRICS_PORT": str(request.param["metrics_port"]),

        "FASTAPI_DOCS_URL": request.param["docs_url"],
        "FASTAPI_OPENAPI_URL": request.param["openapi_url"],
        "FASTAPI_REDOC_URL": request.param["redoc_url"],

        "AUTH_BYPASS_EXTERNAL": str(int(request.param["bypass_auth"])),
        "GITLAB_API_URL": request.param["gitlab_url"],

        "PALM_TEXT_MODEL_NAME": request.param["palm_text_model"],
        "PALM_TEXT_PROJECT": request.param["palm_text_project"],
        "PALM_TEXT_LOCATION": request.param["palm_text_location"],

        "F_THIRD_PARTY_AI_LIMITED_ACCESS": str(tmp_file_limited_access),
        "F_IS_THIRD_PARTY_AI_DEFAULT": str(int(request.param["is_third_party_ai_default"])),
    }

    with mock.patch.dict(os.environ, envs):
        yield


@pytest.mark.parametrize("mock_env_vars", [test_data], indirect=True)
@pytest.mark.parametrize("configuration", [test_data])
def test_config(mock_env_vars, configuration):
    config = Config()

    assert config.triton.host == configuration["triton_host"]
    assert config.triton.port == configuration["triton_port"]

    assert config.fastapi.api_host == configuration["api_host"]
    assert config.fastapi.api_port == configuration["api_port"]
    assert config.fastapi.metrics_host == configuration["metrics_host"]
    assert config.fastapi.metrics_port == configuration["metrics_port"]

    assert config.fastapi.docs_url == configuration["docs_url"]
    assert config.fastapi.openapi_url == configuration["openapi_url"]
    assert config.fastapi.redoc_url == configuration["redoc_url"]
    assert config.fastapi.uvicorn_logger is not None

    assert config.auth.bypass == configuration["bypass_auth"]
    assert config.auth.gitlab_api_base_url == configuration["gitlab_url"]

    assert config.palm_text_model.name == configuration["palm_text_model"]
    assert config.palm_text_model.project == configuration["palm_text_project"]
    assert config.palm_text_model.location == configuration["palm_text_location"]

    assert config.feature_flags.limited_access_third_party_ai == configuration["limited_access_third_party_ai"]
    assert config.feature_flags.is_third_party_ai_default == configuration["is_third_party_ai_default"]

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

        assert config.gitlab_codegen_model.real_or_fake == expected
        assert config.palm_text_model.real_or_fake == expected
