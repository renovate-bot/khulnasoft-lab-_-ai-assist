import os

import pytest
from unittest import mock

from codesuggestions import Config

test_data = dict(
    triton_host="localhost",
    triton_port=5000,

    api_host="localhost",
    api_port=8080,
    docs_url="docs",
    openapi_url="openapi",
    redoc_url="redoc",

    bypass_auth=True,
    gitlab_url="gitlab"
)


@pytest.fixture
def mock_env_vars(request):
    envs = {
        "TRITON_HOST": request.param["triton_host"],
        "TRITON_PORT": str(request.param["triton_port"]),

        "FASTAPI_API_HOST": request.param["api_host"],
        "FASTAPI_API_PORT": str(request.param["api_port"]),
        "FASTAPI_DOCS_URL": request.param["docs_url"],
        "FASTAPI_OPENAPI_URL": request.param["openapi_url"],
        "FASTAPI_REDOC_URL": request.param["redoc_url"],

        "AUTH_BYPASS_EXTERNAL": str(int(request.param["bypass_auth"])),
        "GITLAB_API_URL": request.param["gitlab_url"]
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
    assert config.fastapi.docs_url == configuration["docs_url"]
    assert config.fastapi.openapi_url == configuration["openapi_url"]
    assert config.fastapi.redoc_url == configuration["redoc_url"]
    assert config.fastapi.uvicorn_logger is not None

    assert config.auth.bypass == configuration["bypass_auth"]
    assert config.auth.gitlab_api_base_url == configuration["gitlab_url"]
