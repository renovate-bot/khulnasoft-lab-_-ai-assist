import os


class CloudConnectorConfig:
    # pylint: disable=direct-environment-variable-reference
    @property
    def service_name(self) -> str:
        return os.environ["CLOUD_CONNECTOR_SERVICE_NAME"]

    # pylint: enable=direct-environment-variable-reference
