import pytest

from ai_gateway.feature_flags.context import (
    current_feature_flag_context,
    is_feature_enabled,
)


def test_is_feature_enabled_with_empty_context():
    assert not is_feature_enabled("test_feature")


def test_is_feature_enabled_with_feature_in_context():
    current_feature_flag_context.set({"test_feature"})
    assert is_feature_enabled("test_feature")


def test_is_feature_enabled_with_feature_not_in_context():
    current_feature_flag_context.set({"other_feature"})
    assert not is_feature_enabled("test_feature")


def test_is_feature_enabled_with_multiple_features():
    current_feature_flag_context.set({"feature1", "feature2", "feature3"})
    assert is_feature_enabled("feature2")
    assert not is_feature_enabled("feature4")


def test_current_feature_flag_context_default():
    # Reset the context to its default value
    current_feature_flag_context.set(set())
    assert current_feature_flag_context.get() == set()


def test_current_feature_flag_context_set_and_get():
    test_flags = {"flag1", "flag2"}
    current_feature_flag_context.set(test_flags)
    assert current_feature_flag_context.get() == test_flags


@pytest.fixture(autouse=True)
def reset_context():
    # This fixture will reset the context before and after each test
    current_feature_flag_context.set([])
    yield
    current_feature_flag_context.set([])
