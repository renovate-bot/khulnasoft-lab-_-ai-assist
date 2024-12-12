import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from q_developer_boto3 import boto3 as q_boto3

from ai_gateway.api.v1 import api_router
from ai_gateway.config import Config
from ai_gateway.integrations.amazon_q.errors import AccessDeniedExceptionReason

q_client = q_boto3.client("q", region_name="us-west-1")


def make_boto_client_exception(
    operation_name: str,
    reason: str,
    message: str,
    error_code: str = "AccessDeniedException",
):
    parsed_response = {
        "Error": {
            "Code": error_code,
            "Message": f"{message}",
        },
        "ResponseMetadata": {
            "HTTPHeaders": {
                "connection": "keep-alive",
                "content-length": "252",
                "content-type": "application/x-amz-json-1.0",
                "date": "Wed, 30 Oct 2024 14:37:22 GMT",
                "x-amzn-requestid": "53e3c5df-99a7-4734-a3da-d067462af19c",
            },
            "HTTPStatusCode": 400,
            "RequestId": "53e3c5df-99a7-4734-a3da-d067462af19c",
            "RetryAttempts": 0,
        },
        "message": f"{message}",
    }

    if reason:
        parsed_response["reason"] = reason

    error_class = q_client.exceptions.from_code(error_code)
    return error_class(parsed_response, operation_name)


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


def merge_request_event_payload():
    return {
        "command": "foo",
        "source": "merge_request",
        "project_path": "PROJ-123",
        "project_id": "PROJ-123",
        "note_id": "NOTE-1234",
        "discussion_id": "1234567890",
        "merge_request_id": "MR-123",
        "merge_request_iid": "IMR-123",
        "source_branch": "source-branch",
        "target_branch": "target-branch",
        "last_commit_id": "12345678",
    }


def issue_event_payload():
    return {
        "command": "foo",
        "source": "issue",
        "project_path": "PROJ-123",
        "project_id": "PROJ-123",
        "note_id": "NOTE-1234",
        "discussion_id": "1234567890",
        "issue_id": "ISS-123",
        "issue_iid": "ISS-123",
    }


def perform_request(mock_client, payload):
    return mock_client.post(
        "/amazon_q/events",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-GitLab-Instance-Id": "47474",
            "X-GitLab-Realm": "self-managed",
            "X-Gitlab-Global-User-Id": "1",
        },
        json={
            "role_arn": "arn:aws:iam::123456789012:role/q-dev-role",
            "code": "code-123",
            "payload": payload,
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
        claims=UserClaims(scopes=[GitLabUnitPrimitive.AMAZON_Q_INTEGRATION]),
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
        response = perform_request(mock_client, merge_request_event_payload())

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": "Unauthorized to perform action"}


class TestEvents:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            global_user_id="1",
            claims=UserClaims(scopes=[GitLabUnitPrimitive.AMAZON_Q_INTEGRATION]),
        )

    def test_failed_aws_creds(
        self,
        mock_client,
        mock_boto3,
        mock_sts_client,
        mock_glgo,
    ):

        boto_error = make_boto_client_exception(
            operation_name="SendEvent",
            reason="",
            message="Credentials expired",
        )

        mock_sts_client.assume_role_with_web_identity.side_effect = boto_error

        response = perform_request(mock_client, issue_event_payload())

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {
            "detail": "An error occurred (AccessDeniedException) when calling the SendEvent operation: Credentials expired"
        }

    def test_wrong_payload_format(
        self,
        mock_client,
        mock_boto3,
        mock_sts_client,
        mock_glgo,
    ):
        payload = issue_event_payload()
        payload.pop("issue_id")
        response = perform_request(mock_client, payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert response.json()["detail"][0]["msg"] == "Field required"

    @pytest.mark.parametrize(
        ("payload", "reason", "auth_grant_method"),
        [
            (
                merge_request_event_payload(),
                AccessDeniedExceptionReason.GITLAB_INVALID_IDENTITY,
                "update_auth_grant",
            ),
            (
                issue_event_payload(),
                AccessDeniedExceptionReason.GITLAB_EXPIRED_IDENTITY,
                "create_auth_grant",
            ),
        ],
    )
    def test_failed_events_call_due_to_access_denied(
        self,
        mock_client,
        mock_boto3,
        mock_q_boto3,
        mock_sts_client,
        mock_glgo,
        credentials,
        payload,
        reason,
        auth_grant_method,
    ):
        access_denied_exception = make_boto_client_exception(
            operation_name="SendEvent",
            reason=reason,
            message="Invalid Identity",
        )

        mock_send_event = MagicMock(return_value=None)
        mock_send_event.side_effect = [access_denied_exception, None]

        mock_q_client_response = MagicMock()
        mock_q_client_response.send_event = mock_send_event

        mock_auth_grant = MagicMock(return_value=None)
        setattr(mock_q_client_response, auth_grant_method, mock_auth_grant)

        mock_q_boto3.client.return_value = mock_q_client_response

        response = perform_request(mock_client, payload)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_q_boto3.client.assert_called_once_with("q", **credentials)

        send_event_call = call(
            providerId="GITLAB",
            eventId="Quick Action",
            eventVersion="1.0",
            event=json.dumps(payload, separators=(",", ":")),
        )

        mock_send_event.assert_has_calls([send_event_call, send_event_call])
        mock_auth_grant.assert_called_once_with(code="code-123")

    def test_failed_events_call_due_to_other_error(
        self,
        mock_client,
        mock_boto3,
        mock_q_boto3,
        mock_sts_client,
        mock_glgo,
        credentials,
    ):
        other_error = make_boto_client_exception(
            error_code="ValidationException",
            operation_name="SendEvent",
            reason="other",
            message="Invalid Identity",
        )

        mock_send_event = MagicMock(return_value=None)
        mock_send_event.side_effect = other_error

        mock_q_client_response = MagicMock()
        mock_q_client_response.send_event = mock_send_event
        mock_q_boto3.client.return_value = mock_q_client_response

        response = perform_request(mock_client, merge_request_event_payload())

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.parametrize(
        "payload", [merge_request_event_payload(), issue_event_payload()]
    )
    def test_successful_events_call(
        self,
        mock_client,
        mock_boto3,
        mock_q_boto3,
        mock_sts_client,
        mock_glgo,
        credentials,
        payload,
    ):
        mock_send_event = MagicMock(return_value=None)
        mock_q_client_response = MagicMock()
        mock_q_client_response.send_event = mock_send_event

        mock_q_boto3.client.return_value = mock_q_client_response

        response = perform_request(mock_client, payload)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_q_boto3.client.assert_called_once_with("q", **credentials)

        mock_send_event.assert_called_once_with(
            providerId="GITLAB",
            eventId="Quick Action",
            eventVersion="1.0",
            event=json.dumps(payload, separators=(",", ":")),
        )
