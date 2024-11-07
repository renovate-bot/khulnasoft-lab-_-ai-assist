import pytest

from ai_gateway.cloud_connector import CloudConnectorConfig, NoServiceNameError


@pytest.mark.parametrize(
    (
        "environment_variable_set",
        "service_name_set",
        "error_expected",
    ),
    [
        (
            True,
            True,
            False,
        ),
        (
            True,
            False,
            False,
        ),
        (
            False,
            True,
            False,
        ),
        (
            False,
            False,
            True,
        ),
    ],
)
def test_cloud_connector_config(
    environment_variable_set, service_name_set, error_expected, monkeypatch
):
    monkeypatch.delenv("CLOUD_CONNECTOR_SERVICE_NAME", raising=False)
    CloudConnectorConfig.set_service_name("")

    if environment_variable_set:
        monkeypatch.setenv("CLOUD_CONNECTOR_SERVICE_NAME", "test_service_name")
    if service_name_set:
        CloudConnectorConfig().set_service_name("test_service_name")

    if error_expected:
        with pytest.raises(NoServiceNameError) as ex:
            _result = CloudConnectorConfig().service_name

        assert (
            str(ex.value)
            == "Either CLOUD_CONNECTOR_SERVICE_NAME environment variable should be set or CloudConnectorConfig.set_service_name should be configured."
        )
    else:
        try:
            assert CloudConnectorConfig().service_name == "test_service_name"
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
