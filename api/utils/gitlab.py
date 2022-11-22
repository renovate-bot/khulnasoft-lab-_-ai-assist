import urllib.parse
import logging

import requests

from .cache import Cache


class GitLab:
    def __init__(self, bypass: bool, base_url):
        self.api_base_url = base_url
        self.cache = Cache()
        self.bypass = bypass
        if self.bypass is True:
            logging.critical("EXTERNAL AUTH IS DISABLED, ALL USERS ALLOWED")

    def check_api(self, token: str) -> bool:
        """
        Checks against the GitLab API if a user is allowed to use AI Assist
        :param token:
        :return:
        """
        end_point = "ml/ai-assist"
        url = urllib.parse.urljoin(self.api_base_url, end_point)
        headers = dict(Authorization=f"Bearer {token}")
        try:
            r = requests.get(url=url, headers=headers)
            if r.status_code == 200:  # TODO: Perform a better check
                if r.json().get("user_is_allowed", False) is True:
                    return True
        except requests.exceptions.ConnectionError:
            logging.error("Unable to connect to API for external auth")
            pass
        logging.error("Auth failed")
        return False

    def user_is_allowed(self, token: str) -> bool:
        """
        Main function that checks if a user is allowed to use AI Assist
        :param token: Users Personal Access Token or OAuth token
        :return: bool
        """
        if self.bypass is True:
            logging.debug("External auth is disabled, allowing user")
            return True

        if self.cache.get_cached_token(key=token) is True:
            return True

        checked_value = self.check_api(token=token)
        self.cache.set_cached_token(key=token, value=checked_value)
        return checked_value
