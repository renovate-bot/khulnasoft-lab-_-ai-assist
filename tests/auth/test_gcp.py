import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from google.auth.credentials import Credentials, TokenState

from ai_gateway.auth.gcp import _fetch_application_default_credentials, access_token


class TestAccessToken(unittest.TestCase):
    def test_access_token_new_token(self):
        with patch(
            "ai_gateway.auth.gcp._fetch_application_default_credentials"
        ) as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="new_token")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.FRESH)
            mock_fetch.return_value = mock_creds

            token = access_token()

            mock_fetch.assert_called_once()
            assert token == "new_token"

    def test_access_token_expired_token(self):
        with patch(
            "ai_gateway.auth.gcp._fetch_application_default_credentials"
        ) as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="expired_token")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.STALE)
            mock_fetch.return_value = mock_creds

            access_token()

            self.assertEqual(mock_fetch.call_count, 2)


class TestFetchApplicationDefaultCredentials(unittest.TestCase):
    def test_fetch_application_default_credentials(self):
        with patch("google.auth.default") as mock_default:
            mock_creds = MagicMock(spec=Credentials)
            mock_default.return_value = (mock_creds, None)

            _fetch_application_default_credentials()

            mock_creds.refresh.assert_called_once()
