import pytest

from ai_gateway.agents.base import Agent


@pytest.fixture
def prompt_template() -> dict[str, str]:
    return {"system": "Hi, I'm {{name}}", "user": "{{content}}"}


class TestAgent:
    def test_build_messages(self, prompt_template):
        messages = Agent.build_messages(
            prompt_template, {"name": "Duo", "content": "What's up?"}
        )

        assert messages == [("system", "Hi, I'm Duo"), ("user", "What's up?")]
