import pytest
from langchain_core.runnables import chain
from pydantic.v1.error_wrappers import ValidationError

from ai_gateway.agents.base import Agent
from ai_gateway.gitlab_features import GitLabUnitPrimitive


@pytest.fixture
def prompt_template() -> dict[str, str]:
    return {"system": "Hi, I'm {{name}}", "user": "{{content}}"}


class TestAgent:
    def test_initialize(self):
        @chain
        def runnable(): ...

        agent = Agent(
            name="test", chain=runnable, unit_primitives=["analyze_ci_job_failure"]
        )

        assert agent.name == "test"
        assert agent.bound == runnable  # pylint: disable=comparison-with-callable
        assert agent.unit_primitives == [GitLabUnitPrimitive.ANALYZE_CI_JOB_FAILURE]

    def test_invalid_initialize(self):
        @chain
        def runnable(): ...

        with pytest.raises(ValidationError):
            Agent(name="test", chain=runnable, unit_primitives=["invalid"])

    def test_build_messages(self, prompt_template):
        messages = Agent.build_messages(
            prompt_template, {"name": "Duo", "content": "What's up?"}
        )

        assert messages == [("system", "Hi, I'm Duo"), ("user", "What's up?")]
