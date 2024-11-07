import os


class NoServiceNameError(Exception):
    pass


class CloudConnectorConfig:
    _service_name = None

    @classmethod
    def set_service_name(cls, name: str):
        cls._service_name = name

    @property
    def service_name(self) -> str:
        # pylint: disable=direct-environment-variable-reference
        service_name = (
            os.environ.get("CLOUD_CONNECTOR_SERVICE_NAME") or self._service_name
        )
        # pylint: enable=direct-environment-variable-reference

        if service_name:
            return service_name

        raise NoServiceNameError(
            "Either CLOUD_CONNECTOR_SERVICE_NAME environment variable should be set or CloudConnectorConfig.set_service_name should be configured."
        )
