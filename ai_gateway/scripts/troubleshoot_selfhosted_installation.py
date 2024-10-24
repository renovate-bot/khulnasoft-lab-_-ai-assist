#!/usr/bin/env python3
# A script that tests if a connection to the server works

import argparse
import os

import boto3
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


def check_provider_accessible(provider):
    # pylint: disable=direct-environment-variable-reference
    provider_name = provider.capitalize()
    print(f"Testing if provider {provider_name} is accessible ...")

    if provider == "bedrock":
        boto3_bedrock = boto3.client(
            provider,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME"),
        )

        try:
            boto3_bedrock.list_foundation_models()
        except Exception as e:
            print(f": {e}")
            error_message = f"""
                >> An error occurred while contacting provider {provider}: {e}
                >> Potential cause(s) are:
                >> - Access keys are not valid:
                >>   Make sure the following environment variables are set to the correct values:
                >>   - AWS_ACCESS_KEY_ID
                >>   - AWS_SECRET_ACCESS_KEY
                >>   - AWS_REGION_NAME
                >> - Network issues:
                >>   Verify if you can reach the internet from the server.
                """
            raise RuntimeError(error_message)

    else:
        raise ValueError(f"Provider {provider_name} is not a supported provider.")

    print(f">> Provider {provider_name} is accessible ✔\n")


def check_general_env_variables():
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


def check_provider_specific_env_variables(provider):
    # pylint: disable=direct-environment-variable-reference
    provider_name = provider.capitalize()
    missing_vars = []

    if provider == "bedrock":
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
        ]
    else:
        raise ValueError(f"Provider {provider_name} is not a supported provider.")

    print(f"Testing specific environment variables for provider {provider_name} ...")

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        error_message = f"Missing environment variables: {', '.join(missing_vars)} for provider {provider_name}\n"
        raise ValueError(error_message)
    print(
        f">> Specific environment variables for provider {provider_name} are set correctly ✔\n"
    )


def check_suggestions_model_access(
    endpoint, model_name, model_endpoint, model_key, model_identifier
):
    print(f"Testing if model {model_name} is accessible for Code Generation ...")

    url = f"http://{endpoint}/v2/code/generations"

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
        "model_endpoint": model_endpoint,
        "model_api_key": model_key,
        "model_identifier": model_identifier,
        "model_name": model_name,
        "telemetry": [],
        "stream": False,
        "choices_count": 0,
        "context": [],
        "agent_id": "string",
        "prompt_id": "code_suggestions/generations",
        "prompt_version": 2,
        "prompt": "",
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


def troubleshoot():
    # Parse the endpoint, model_name, model_endpoint, model_key, model_identifier with argparse

    parser = argparse.ArgumentParser(description="Test AI Gateway and model access")
    parser.add_argument(
        "--endpoint", default="localhost:5052", help="AI Gateway endpoint"
    )
    parser.add_argument(
        "--model-name", required=False, help="Name of the model to test"
    )
    parser.add_argument(
        "--model-endpoint",
        required=False,
        default="http://localhost:4000",
        help="Endpoint of the model. Example: http://localhost:4000. "
        "When using a model from an online provider like Bedrock, "
        "this can be left empty.",
    )
    parser.add_argument(
        "--model-identifier", required=False, help="Identifier of the model"
    )
    parser.add_argument("--model-key", required=False, help="API key for the model")

    args = parser.parse_args()

    # Check if model_endpoint is required but not provided
    if args.model_identifier is None and args.model_endpoint is None:
        parser.error(
            "--model-endpoint is required when --model-identifier is not provided"
        )

    # If model_endpoint is not provided, set a default value
    if args.model_endpoint is None:
        args.model_endpoint = "http://localhost:4000"

    endpoint = args.endpoint
    model_name = args.model_name
    model_endpoint = args.model_endpoint
    model_identifier = args.model_identifier
    model_key = args.model_key

    # if model_identifier is provided, extract the provider
    # example: extract `bedrock` from `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0`
    provider = model_identifier.split("/")[0] if model_identifier else None

    check_general_env_variables()
    check_aigw_endpoint(endpoint)

    if model_name:
        if provider:
            check_provider_specific_env_variables(provider)
            check_provider_accessible(provider)

        check_suggestions_model_access(
            endpoint, model_name, model_endpoint, model_key, model_identifier
        )
