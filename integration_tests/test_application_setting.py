import os


def test_application_setting():
    """
    Ensure application settings are set correctly for integration tests.
    """
    # pylint: disable=direct-environment-variable-reference
    assert os.getenv("AIGW_AUTH__BYPASS_EXTERNAL", "false") == "false"
    assert os.getenv("AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER", "false") == "true"
    # pylint: enable=direct-environment-variable-reference
