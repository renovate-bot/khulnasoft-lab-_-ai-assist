import pytest
from langchain.tools import Tool

from autograph.entities import AgentConfig


@pytest.fixture
def agent_config():
    return AgentConfig(
        model="davinci",
        name="test agent",
        tools=["search", "calculator"],
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        goal="Provide a helpful response.",
    )


@pytest.fixture
def tools():
    return [
        Tool(
            name="search",
            description="Search the web for information",
            func=(lambda x: x),
        ),
        Tool(
            name="calculator",
            description="Perform mathematical calculations",
            func=(lambda x: x),
        ),
    ]
