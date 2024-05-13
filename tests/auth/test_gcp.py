import time
import unittest
from unittest.mock import patch

from ai_gateway.auth.gcp import access_token


class TestAccessToken(unittest.TestCase):
    def test_access_token_new_token(self):
        with patch("ai_gateway.auth.gcp._fetch_access_token_from_adr") as mock_fetch:
            mock_fetch.return_value = ("new_token", time.time())

            token = access_token()

            mock_fetch.assert_called_once()
            assert token == "new_token"

    def test_access_token_expired_token(self):
        with patch("ai_gateway.auth.gcp._fetch_access_token_from_adr") as mock_fetch:
            created_at = time.time() - 3600  # an hour ago
            mock_fetch.return_value = ("expired_token", created_at)

            access_token()

            self.assertEqual(mock_fetch.call_count, 2)
