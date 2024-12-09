from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException, status

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.auth.glgo import GlgoAuthority
from ai_gateway.integrations.amazon_q.client import AmazonQClient, AmazonQClientFactory
from ai_gateway.integrations.amazon_q.errors import AWSException


class TestAmazonQClientFactory:
    @pytest.fixture
    def mock_glgo_authority(self):
        return MagicMock(spec=GlgoAuthority)

    @pytest.fixture
    def mock_sts_client(self):
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_boto3(self, mock_sts_client):
        with patch("ai_gateway.integrations.amazon_q.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sts_client
            yield mock_boto3

    @pytest.fixture
    def amazon_q_client_factory(self, mock_glgo_authority, mock_boto3):
        return AmazonQClientFactory(
            glgo_authority=mock_glgo_authority,
            endpoint_url="https://mock.endpoint",
            region="us-east-1",
        )

    @pytest.fixture
    def mock_user(self):
        user = MagicMock(spec=StarletteUser)
        user.global_user_id = "test-user-id"
        user.claims = MagicMock(subject="test-session")
        return user

    def test_get_glgo_token(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_glgo_authority.token.return_value = "mock-token"
        token = amazon_q_client_factory._get_glgo_token(
            mock_user, "Bearer mock-cloud-connector-token"
        )

        mock_glgo_authority.token.assert_called_once_with(
            user_id="test-user-id", cloud_connector_token="mock-cloud-connector-token"
        )
        assert token == "mock-token"

    def test_missing_user_id_for_glgo_token(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_user.global_user_id = None

        with pytest.raises(HTTPException) as exc:
            amazon_q_client_factory._get_glgo_token(
                mock_user, "Bearer mock-cloud-connector-token"
            )
        assert exc.value.status_code == 400
        assert exc.value.detail == "User Id is missing"

    def test_glgo_token_raises_error(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority
    ):
        mock_glgo_authority.token.side_effect = KeyError()

        with pytest.raises(HTTPException) as exc:
            amazon_q_client_factory._get_glgo_token(
                mock_user, "Bearer mock-cloud-connector-token"
            )
        assert exc.value.status_code == 500
        assert exc.value.detail == "Cannot obtain OIDC token"

    def test_get_aws_credentials(
        self, amazon_q_client_factory, mock_user, mock_sts_client
    ):
        mock_sts_client.assume_role_with_web_identity.return_value = {
            "Credentials": {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }
        }

        credentials = amazon_q_client_factory._get_aws_credentials(
            mock_user, token="mock-web-identity-token", role_arn="mock-role-arn"
        )

        mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
            RoleArn="mock-role-arn",
            RoleSessionName="test-session",
            WebIdentityToken="mock-web-identity-token",
            DurationSeconds=43200,
        )
        assert credentials == {
            "AccessKeyId": "mock-key",
            "SecretAccessKey": "mock-secret",
            "SessionToken": "mock-token",
        }

    def test_get_aws_credentials_no_claims(
        self, amazon_q_client_factory, mock_user, mock_sts_client
    ):
        mock_user.claims = None
        mock_sts_client.assume_role_with_web_identity.return_value = {
            "Credentials": {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }
        }

        credentials = amazon_q_client_factory._get_aws_credentials(
            mock_user, token="mock-web-identity-token", role_arn="mock-role-arn"
        )

        mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
            RoleArn="mock-role-arn",
            RoleSessionName="placeholder",
            WebIdentityToken="mock-web-identity-token",
            DurationSeconds=43200,
        )

        assert credentials == {
            "AccessKeyId": "mock-key",
            "SecretAccessKey": "mock-secret",
            "SessionToken": "mock-token",
        }

    def test_get_client(
        self, amazon_q_client_factory, mock_user, mock_glgo_authority, mock_sts_client
    ):
        with patch(
            "ai_gateway.integrations.amazon_q.client.AmazonQClient"
        ) as mock_q_client_class:
            mock_q_client_instance = MagicMock()
            mock_q_client_class.return_value = mock_q_client_instance

            credentials = {
                "AccessKeyId": "mock-key",
                "SecretAccessKey": "mock-secret",
                "SessionToken": "mock-token",
            }

            mock_glgo_authority.token.return_value = "mock-token"
            mock_sts_client.assume_role_with_web_identity.return_value = {
                "Credentials": credentials
            }

            client = amazon_q_client_factory.get_client(
                current_user=mock_user,
                auth_header="Bearer mock-cloud-connector-token",
                role_arn="mock-role-arn",
            )

            mock_glgo_authority.token.assert_called_once_with(
                user_id="test-user-id",
                cloud_connector_token="mock-cloud-connector-token",
            )

            mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
                RoleArn="mock-role-arn",
                RoleSessionName="test-session",
                WebIdentityToken="mock-token",
                DurationSeconds=43200,
            )

            mock_q_client_class.assert_called_once_with(
                url="https://mock.endpoint", region="us-east-1", credentials=credentials
            )

            assert client == mock_q_client_instance


class TestAmazonQClient:
    @pytest.fixture
    def mock_credentials(self):
        return {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
        }

    @pytest.fixture
    def mock_application_request(self):
        class ApplicationRequest:
            client_id = "test-client-id"
            client_secret = "test-secret"
            instance_url = "https://test.example.com"
            redirect_url = "https://test.example.com/callback"

        return ApplicationRequest()

    @pytest.fixture
    def mock_q_client(self):
        with patch(
            "ai_gateway.integrations.amazon_q.client.q_boto3.client"
        ) as mock_client:
            yield mock_client.return_value

    @pytest.fixture
    def q_client(self, mock_credentials, mock_q_client):
        return AmazonQClient(
            url="https://q-api.example.com",
            region="us-west-2",
            credentials=mock_credentials,
        )

    @pytest.fixture
    def params(self):
        return dict(
            clientId="test-client-id",
            clientSecret="test-secret",
            instanceUrl="https://test.example.com",
            redirectUrl="https://test.example.com/callback",
        )

    def test_init_creates_client_with_correct_params(self, mock_credentials):
        with patch(
            "ai_gateway.integrations.amazon_q.client.q_boto3.client"
        ) as mock_client:
            AmazonQClient(
                url="https://q-api.example.com",
                region="us-west-2",
                credentials=mock_credentials,
            )

            mock_client.assert_called_once_with(
                "q",
                region_name="us-west-2",
                endpoint_url="https://q-api.example.com",
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_session_token="test-session-token",
            )

    def test_create_auth_application_success(
        self, q_client, mock_q_client, mock_application_request, params
    ):
        q_client.create_or_update_auth_application(mock_application_request)
        mock_q_client.create_o_auth_app_connection.assert_called_once_with(**params)

        assert not mock_q_client.update_o_auth_app_connection.called

    def test_update_auth_application_on_conflict(
        self, q_client, mock_q_client, mock_application_request, params
    ):
        error_response = {
            "Error": {"Code": "ConflictException", "Message": "A conflict occurred"}
        }
        mock_q_client.create_o_auth_app_connection.side_effect = ClientError(
            error_response, "create_o_auth_app_connection"
        )

        q_client.create_or_update_auth_application(mock_application_request)

        mock_q_client.create_o_auth_app_connection.assert_called_once_with(**params)
        mock_q_client.update_o_auth_app_connection.assert_called_once_with(**params)

    def test_raises_non_conflict_aws_errors(
        self, q_client, mock_q_client, mock_application_request
    ):
        error_response = {
            "Error": {"Code": "ValidationException", "Message": "invalid message"}
        }
        mock_q_client.create_o_auth_app_connection.side_effect = ClientError(
            error_response, "create_o_auth_app_connection"
        )

        with pytest.raises(AWSException):
            q_client.create_or_update_auth_application(mock_application_request)

        mock_q_client.create_o_auth_app_connection.assert_called_once()
        assert not mock_q_client.update_o_auth_app_connection.called
