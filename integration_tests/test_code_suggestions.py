import requests

from integration_tests.utils import ai_gateway_url, get_user_jwt


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

    response = requests.post(api_url, headers=headers, json=data, timeout=60)

    return response


def test_code_suggestion_for_gitlab_com():
    token = get_user_jwt("saas")
    response = send_and_check_completion_request(token, "gitlab.com", "saas")

    assert (
        response.status_code == 200
    ), f"/v2/code/completions request failed with status: {response.status_code}, body: {response.text}"


def test_code_suggestion_for_self_managed():
    token = get_user_jwt("self-managed")
    response = send_and_check_completion_request(
        token, "gitlab.company.name", "self-managed"
    )

    assert (
        response.status_code == 200
    ), f"/v2/code/completions request failed with status: {response.status_code}, body: {response.text}"
