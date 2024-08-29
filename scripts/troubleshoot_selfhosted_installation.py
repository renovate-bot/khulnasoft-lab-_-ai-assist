# A script that tests if a connection to the server works

import argparse
import os

import requests


def check_aigw_endpoint(endpoint="localhost:5052"):
    print("Testing if AI Gateway is running...")

    aigw_url = f"http://{endpoint}/monitoring/healthz"

    try:
        health_response = requests.get(
            aigw_url, headers={"accept": "application/json"}, timeout=30
        )

        if health_response.status_code == 200:
            print(">> AI Gateway is up and running ✔\n")
            return

        error_message = (
            f"AI Gateway is not running. Status code: {health_response.status_code}. "
            "Restart the server and verify the logs for more information."
        )
    except (requests.RequestException, ConnectionRefusedError):
        error_message = "AI Gateway is not running. Restart the server, and verify the logs for more information."

    raise RuntimeError(error_message)


def check_env_variables():
    # pylint: disable=direct-environment-variable-reference
    print("Testing environment variables ...")

    if not os.getenv("AIGW_CUSTOM_MODELS__ENABLED", "") == "true":
        raise ValueError(">> AIGW_CUSTOM_MODELS__ENABLED must be set to true")

    if var := os.getenv("AIGW_GITLAB_URL", None):
        print(f">> AIGW_GITLAB_URL is set to {var}")
    else:
        if not os.getenv("AIGW_AUTH__BYPASS_EXTERNAL", "") == "true":
            raise ValueError(
                ">> Either AIGW_GITLAB_URL (preferred) or AIGW_AUTH__BYPASS_EXTERNAL must be set"
            )

        print(
            ">> AIGW_AUTH__BYPASS_EXTERNAL is set to true. This disables authentication, "
            "prefer using AIGW_GITLAB_URL"
        )

    print(">> Env variables are set correctly ✔\n")


def check_suggestions_model_access(endpoint, model_name, model_endpoint, model_key):
    print(f"Testing if model {model_name} is accessible")

    url = f"http://{endpoint}/v2/code/completions"

    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "project_path": "string",
        "project_id": 0,
        "current_file": {
            "file_name": "test.py",
            "language_identifier": "python",
            "content_above_cursor": "def hello():\n  ",
            "content_below_cursor": "",
        },
        "model_provider": "litellm",
        "model_endpoint": "http://localhost:4000",
        "model_api_key": "",
        "model_name": "codegemma_7b",
        "telemetry": [],
        "stream": False,
        "choices_count": 1,
        "context": [],
        "agent_id": "string",
        "prompt_version": 2,
    }

    error_message = f"""
                >> Failed to access {model_name} model."
                >> Potential causes are:
                >> - The model is not running. Verify if your model is running
                >> - Model, model endpoint or the api key are invalid. Double check the parameters passed:
                >>    - model: {model_name}, model endpoint: {model_endpoint}, model key: {model_key}
                >> - The model is not reachable by AI Gateway. This can happen if the network is not configured correctly.
                >>    - Attempt to make a request to your model api directly from the AI Gateway container
                """

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print(f">> Successfully accessed {model_name} model ✔\n")
            return

        error_message = f"""
            {error_message}
            >>
            >> status code: {response.status_code}
            >> status code: {response.text}
            """

    except requests.RequestException as e:
        error_message = f"""
                    {error_message}
                    >>
                    >> error: {e}
                    """

    raise RuntimeError(error_message)


if __name__ == "__main__":
    # PArse the endpoint, model_name, model_endpoint, model_key with argparse

    parser = argparse.ArgumentParser(description="Test AI Gateway and model access")
    parser.add_argument(
        "--endpoint", default="localhost:5052", help="AI Gateway endpoint"
    )
    parser.add_argument("--model-name", required=True, help="Name of the model to test")
    parser.add_argument("--model-endpoint", required=True, help="Endpoint of the model")
    parser.add_argument("--model-key", required=False, help="API key for the model")

    args = parser.parse_args()

    endpoint = args.endpoint
    model_name = args.model_name
    model_endpoint = args.model_endpoint
    model_key = args.model_key

    check_env_variables()
    check_aigw_endpoint(endpoint)
    check_suggestions_model_access(endpoint, model_name, model_endpoint, model_key)
