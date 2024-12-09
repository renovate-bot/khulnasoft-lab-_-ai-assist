import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import botocore
import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.api.v1 import api_router
from ai_gateway.config import Config


def create_access_denied_error():
    error_response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "The user does not have access",
        },
        "ResponseMetadata": {
            "HTTPHeaders": {
                "HTTPStatusCode": 400,
                "RequestId": "cef34aa6-ce28-4b6f-a159-a89fad215348",
                "RetryAttempts": 0,
            },
        },
    }
    return botocore.exceptions.ClientError(error_response, "operation_name")


def create_param_validation_error():
    error_details = (
        "Invalid length for parameter RoleArn, value: 4, valid min length: 20"
    )

    return botocore.exceptions.ParamValidationError(report=error_details)


def create_conflict_exception_error():
    error_response = {
        "Error": {
            "Code": "ConflictException",
            "Message": "This application conflicted with an existing one.",
        }
    }
    return botocore.exceptions.ClientError(
        error_response, "create_o_auth_app_connection"
    )


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


def perform_request(mock_client):
    return mock_client.post(
        "/amazon_q/oauth/application",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-GitLab-Instance-Id": "47474",
            "X-GitLab-Realm": "self-managed",
            "X-Gitlab-Global-User-Id": "1",
        },
        json={
            "client_id": "123",
            "client_secret": "some.secret",
            "instance_url": "https://example.com",
            "redirect_url": "https://example.com",
            "role_arn": "arn:aws:iam::123456789012:role/q-dev-role",
        },
    )


@pytest.fixture
def mock_glgo():
    with patch("requests.post") as requests_mock:
        requests_mock.json.return_value = {"token": "glgo valid token"}
        requests_mock.raise_for_status.return_value = None

        yield requests_mock


@pytest.fixture
def credentials():
    return {
        "region_name": "us-west-1",
        "endpoint_url": "http://example.com",
        "aws_access_key_id": "mock access key id",
        "aws_secret_access_key": "mock secret access key",
        "aws_session_token": "mock session token",
    }


@pytest.fixture
def mock_sts_client(credentials):
    mock_client = MagicMock()
    mock_client.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": credentials["aws_access_key_id"],
            "SecretAccessKey": credentials["aws_secret_access_key"],
            "SessionToken": credentials["aws_session_token"],
            "Expiration": datetime.now() + timedelta(days=1),
        }
    }

    return mock_client


@pytest.fixture
def mock_boto3(mock_sts_client):
    with patch("ai_gateway.integrations.amazon_q.client.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sts_client
        yield mock_boto3


@pytest.fixture
def mock_q_boto3():
    with patch("ai_gateway.integrations.amazon_q.client.q_boto3") as mock_q_boto3:
        yield mock_q_boto3


@pytest.fixture
def auth_user(self):
    return CloudConnectorUser(
        authenticated=True,
        global_user_id="1",
        claims=UserClaims(scopes=[GitLabUnitPrimitive.AGENT_QUICK_ACTIONS]),
    )


@pytest.fixture
def mock_config(assets_dir, credentials):
    config = Config()
    config.amazon_q.endpoint_url = credentials["endpoint_url"]
    config.amazon_q.region = credentials["region_name"]
    config.self_signed_jwt.signing_key = open(
        assets_dir / "keys" / "signing_key.pem"
    ).read()

    yield config


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            global_user_id="1",
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client):
        response = perform_request(mock_client)

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": "Unauthorized to perform action"}


class TestApplication:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            global_user_id="1",
            claims=UserClaims(scopes=[GitLabUnitPrimitive.AGENT_QUICK_ACTIONS]),
        )

    @pytest.mark.parametrize(
        (
            "boto_error",
            "expected_status",
            "expected_msg",
        ),
        [
            (
                create_access_denied_error(),
                status.HTTP_403_FORBIDDEN,
                "An error occurred (AccessDeniedException) when calling the operation_name operation: The user does not have access",
            ),
            (
                create_param_validation_error(),
                status.HTTP_400_BAD_REQUEST,
                "Parameter validation failed:\nInvalid length for parameter RoleArn, value: 4, valid min length: 20",
            ),
        ],
    )
    def test_failed_aws_creds(
        self,
        mock_client,
        mock_boto3,
        mock_sts_client,
        mock_glgo,
        boto_error,
        expected_status,
        expected_msg,
    ):
        mock_sts_client.assume_role_with_web_identity.side_effect = boto_error

        response = perform_request(mock_client)

        assert response.status_code == expected_status
        assert response.json() == {"detail": expected_msg}

    def test_successful_oauth_application(
        self,
        mock_client,
        mock_boto3,
        mock_q_boto3,
        mock_sts_client,
        mock_glgo,
        credentials,
    ):
        mock_create_o_auth_app_connection = MagicMock(return_value=None)
        mock_q_client_response = MagicMock()
        mock_q_client_response.create_o_auth_app_connection = (
            mock_create_o_auth_app_connection
        )

        mock_q_boto3.client.return_value = mock_q_client_response

        response = perform_request(mock_client)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_q_boto3.client.assert_called_once_with("q", **credentials)

        mock_create_o_auth_app_connection.assert_called_once_with(
            clientId="123",
            clientSecret="some.secret",
            instanceUrl="https://example.com",
            redirectUrl="https://example.com",
        )

    def test_successful_oauth_application_with_conflict(
        self,
        mock_client,
        mock_boto3,
        mock_q_boto3,
        mock_sts_client,
        mock_glgo,
        credentials,
    ):
        mock_create_o_auth_app_connection = MagicMock(
            side_effect=[create_conflict_exception_error(), None]
        )
        mock_update_o_auth_app_connection = MagicMock()

        mock_q_client_response = MagicMock()
        mock_q_client_response.create_o_auth_app_connection = (
            mock_create_o_auth_app_connection
        )
        mock_q_client_response.update_o_auth_app_connection = (
            mock_update_o_auth_app_connection
        )

        mock_q_boto3.client.return_value = mock_q_client_response

        response = perform_request(mock_client)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_q_boto3.client.assert_called_once_with("q", **credentials)

        params = dict(
            clientId="123",
            clientSecret="some.secret",
            instanceUrl="https://example.com",
            redirectUrl="https://example.com",
        )

        mock_create_o_auth_app_connection.assert_called_once_with(**params)
        mock_update_o_auth_app_connection.assert_called_once_with(**params)
