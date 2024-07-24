from contextvars import ContextVar

import pytest

from ai_gateway.internal_events.context import EventContext, current_event_context


def test_event_context_default_values():
    context = EventContext()
    assert context.environment == "development"
    assert context.source == "ai-gateway-python"
    assert context.realm is None
    assert context.instance_id is None
    assert context.host_name is None
    assert context.instance_version is None
    assert context.global_user_id is None


def test_event_context_custom_values():
    context = EventContext(
        environment="production",
        source="gitlab",
        realm="user",
        instance_id="123",
        host_name="example.com",
        instance_version="14.0.0",
        global_user_id="user123",
    )
    assert context.environment == "production"
    assert context.source == "gitlab"
    assert context.realm == "user"
    assert context.instance_id == "123"
    assert context.host_name == "example.com"
    assert context.instance_version == "14.0.0"
    assert context.global_user_id == "user123"


def test_current_event_context_default():
    assert isinstance(current_event_context, ContextVar)
    assert isinstance(current_event_context.get(), EventContext)
    assert current_event_context.get().environment == "development"


def test_current_event_context_set_and_reset():
    original_context = current_event_context.get()

    new_context = EventContext(environment="staging")
    token = current_event_context.set(new_context)

    assert current_event_context.get().environment == "staging"

    current_event_context.reset(token)
    assert current_event_context.get() == original_context


def test_event_context_model_validation():
    with pytest.raises(ValueError):
        EventContext(environment=123)


def test_event_context_optional_fields():
    context = EventContext(environment="test")
    assert context.environment == "test"
    assert context.source == "ai-gateway-python"
