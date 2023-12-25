from unittest import mock

import pytest

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.x_ray.typing import AnyPromptComponent
from ai_gateway.auth import User, UserClaims
from ai_gateway.deps import XRayContainer
from ai_gateway.models import AnthropicModel, SafetyAttributes, TextGenModelOutput


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["code_suggestions"]),
    )


class TestXRayLibraries:
    @pytest.mark.parametrize(
        (
            "model_output_text",
            "want_called",
            "want_status",
            "want_prompt",
        ),
        [
            (
                '{"libraries": [{"name": "kaminari", "description": "Pagination"}]}',
                True,
                200,
                "Human: Parse Gemfile content: `gem kaminari`. Respond using only valid JSON with list of libraries",
            ),
        ],
    )
    def test_successful_request(
        self,
        mock_client,
        model_output_text,
        want_called,
        want_status,
        want_prompt,
    ):
        model_safety_attr = SafetyAttributes(blocked=False)
        model_output = TextGenModelOutput(
            text=model_output_text,
            score=0,
            safety_attributes=model_safety_attr,
        )

        anthropic_model_mock = mock.Mock(spec=AnthropicModel)
        anthropic_model_mock.generate = mock.AsyncMock(return_value=model_output)
        container = XRayContainer()

        with container.anthropic_model.override(anthropic_model_mock):
            response = mock_client.post(
                "/x-ray/libraries",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json={
                    "prompt_components": [
                        {
                            "type": "x_ray_package_file_prompt",
                            "payload": {
                                "prompt": "Human: Parse Gemfile content: `gem kaminari`. Respond using only valid JSON with list of libraries",
                                "provider": "anthropic",
                                "model": "claude-2.0",
                            },
                            "metadata": {"scannerVersion": "0.0.1"},
                        }
                    ]
                },
            )

        assert response.status_code == want_status
        assert anthropic_model_mock.generate.called == want_called

        if want_called:
            anthropic_model_mock.generate.assert_called_with(
                prefix=want_prompt, _suffix=""
            )

        assert response.json() == {"response": model_output_text}


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    @pytest.mark.parametrize("path", ["/x-ray/libraries"])
    def test_failed_authorization_scope(self, mock_client, path):
        response = mock_client.post(
            path,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt_components": [
                    {
                        "type": "x_ray_package_file_prompt",
                        "payload": {
                            "prompt": "Human: Parse Gemfile content: `gem kaminari`. Respond using only valid JSON with list of libraries",
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                        "metadata": {"scannerVersion": "0.0.1"},
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}


class TestAnyPromptComponent:
    @pytest.mark.parametrize(
        ("size", "want_error"), [(0, False), (10, False), (11, True)]
    )
    def test_metadata_length_validation(self, size, want_error):
        metadata = {f"key{i}": f"value{i}" for i in range(size)}

        if want_error:
            with pytest.raises(ValueError):
                AnyPromptComponent(type="type", payload="{}", metadata=metadata)
        else:
            AnyPromptComponent(type="type", payload="{}", metadata=metadata)
