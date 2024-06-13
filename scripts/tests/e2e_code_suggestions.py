# To run it locally similar to how it runs in CI:
#
# 1. Set `AIGW_AUTH__BYPASS_EXTERNAL=false` and `AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER=true` in `.env`
# The test will always pass if `AIGW_AUTH__BYPASS_EXTERNAL=true` (default) but it will not match what we run in CI.
# The test will always fail if `AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER=false` (default) - we need this set to `true`.
#
# 2. Restart you local AI GW: `poetry run ai_gateway`.
# You need local AI GW up and running during the test.
#
# 3. Run the command below (adjust `AI_GATEWAY_URL` for your local AI GW url):
# `AI_GATEWAY_URL=http://localhost:5052 poetry run python scripts/tests/e2e_code_suggestions.py`
#
# 4. It will run 2 scenarios: for `GitLab.com` and `Self-Managed` against your local AI GW instance.
# Ensure `/v2/code/completions request successful!` for both.

import os
import sys

import requests


# pylint: disable=direct-environment-variable-reference
def ai_gateway_url():
    return os.getenv("AI_GATEWAY_URL", "http://localhost:5052")


# pylint: enable=direct-environment-variable-reference


# Function to get user JWT
def get_user_jwt(realm):
    api_url = f"{ai_gateway_url()}/v1/code/user_access_token"

    headers = {
        "Bypass-Auth": "true",
        "X-Gitlab-Realm": realm,
        "X-Gitlab-Global-User-Id": "111",
    }

    response = requests.post(api_url, headers=headers)
    response.raise_for_status()

    return response.json().get("token")


# Function to send completion request
def send_and_check_completion_request(user_jwt, host_name, realm):
    api_url = f"{ai_gateway_url()}/v2/code/completions"

    headers = {
        "Authorization": f"Bearer {user_jwt}",
        "Content-Type": "application/json",
        "X-Gitlab-Authentication-Type": "oidc",
        "X-Gitlab-Global-User-Id": "111",
        "X-Gitlab-Host-Name": host_name,
        "X-Gitlab-Instance-Id": "ea8bf810-1d6f-4a6a-b4fd-93e8cbd8b57f",
        "X-Gitlab-Realm": realm,
        "X-Gitlab-Version": "17.1.0",
        "User-Agent": "node-fetch (+https://github.com/node-fetch/node-fetch)",
    }

    if realm == "saas":
        headers.update(
            {
                "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "9970,6543",
                "X-Gitlab-Saas-Namespace-Ids": "",
            }
        )

    data = {
        "current_file": {
            "file_name": "test",
            "content_above_cursor": "func hello_world(){\n\t",
            "content_below_cursor": "\n}",
            "stream": False,
        },
        "prompt_version": 1,
        "metadata": {"source": "Gitlab EE", "version": "17.1.0"},
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        print("/v2/code/completions request successful!")
    else:
        print(
            f"/v2/code/completions request failed with status: {response.status_code}, body: {response.text}"
        )
        sys.exit(1)


def test_for_gitlab_com():
    print("Testing GitLab.com (SaaS) flow")
    token = get_user_jwt("saas")
    send_and_check_completion_request(token, "gitlab.com", "saas")


def test_for_self_managed():
    print("Testing Self-Managed flow")
    token = get_user_jwt("self-managed")
    send_and_check_completion_request(token, "gitlab.company.name", "self-managed")


if __name__ == "__main__":
    test_for_gitlab_com()
    test_for_self_managed()
