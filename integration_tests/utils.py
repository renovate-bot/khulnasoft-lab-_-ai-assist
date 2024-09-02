import os

import requests


# pylint: disable=direct-environment-variable-reference
def ai_gateway_url() -> str:
    return os.getenv("AI_GATEWAY_URL", "http://localhost:5052")


# pylint: enable=direct-environment-variable-reference


def get_user_jwt(realm: str) -> str | None:
    api_url = f"{ai_gateway_url()}/v1/code/user_access_token"

    headers = {
        "Bypass-Auth": "true",
        "X-Gitlab-Realm": realm,
        "X-Gitlab-Global-User-Id": "111",
        "X-Gitlab-Instance-Id": "ea8bf810-1d6f-4a6a-b4fd-93e8cbd8b57f",
    }

    response = requests.post(api_url, headers=headers, timeout=60)
    response.raise_for_status()

    return response.json().get("token")
