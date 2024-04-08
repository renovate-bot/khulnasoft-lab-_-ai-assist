import tempfile
from pathlib import Path
from typing import Any

import pytest
from jinja2 import BaseLoader, Environment

from ai_gateway.chat.prompts import ChatPrompt, LocalPromptRegistry

TPL_PROMPT_LLM = "This is a very simple prompt template: {{ text }}"
TPL_PROMPT_CHAT_USER = "This is a very simple user prompt template: {{ text }}"
TPL_PROMPT_CHAT_SYSTEM = "This is a very simple system prompt template: {{ text }}"
TPL_PROMPT_CHAT_AI = "This is a very simple assistant prompt template: {{ text }}"


@pytest.fixture
def tmp_prompt_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def _write_tpl(tmp_prompt_dir: Path, key: str, tpl: str):
    tpl_path = tmp_prompt_dir / f"{key}.jinja"
    with open(str(tpl_path), "w") as fp:
        fp.write(tpl)


def _write_chat_tpls(tmp_prompt_dir: Path, tpls: dict[str, str]):
    for key, tpl in tpls.items():
        _write_tpl(tmp_prompt_dir, key, tpl)


def _render_tpl(tpl: str, **kwargs: Any) -> str:
    env_jinja = Environment(loader=BaseLoader()).from_string(tpl)
    return env_jinja.render(**kwargs)


def _render_chat_tpl(tpls: dict[str, str], **kwargs) -> ChatPrompt:
    data = {
        key: Environment(loader=BaseLoader()).from_string(tpl).render(**kwargs)
        for key, tpl in tpls.items()
    }
    return ChatPrompt(**data)


class TestLocalPromptRegistry:
    @pytest.mark.parametrize("text", [None, "word1_word2"])
    def test_llm(self, tmp_prompt_dir: Path, text: str | None):
        _write_tpl(tmp_prompt_dir, "tpl", TPL_PROMPT_LLM)

        registry = LocalPromptRegistry.from_resources(
            {
                "llm": tmp_prompt_dir,
            }
        )

        kwargs = {"text": text} if text else {}
        actual = registry.get_prompt("llm", **kwargs)
        expected = _render_tpl(TPL_PROMPT_LLM, **kwargs)

        assert actual == expected

    @pytest.mark.parametrize(
        ("tpls", "text"),
        [
            ({"user": TPL_PROMPT_CHAT_USER}, None),
            ({"user": TPL_PROMPT_CHAT_USER}, "word1_word2"),
            (
                {"system": TPL_PROMPT_CHAT_SYSTEM, "user": TPL_PROMPT_CHAT_USER},
                "word1_word2",
            ),
            (
                {
                    "assistant": TPL_PROMPT_CHAT_AI,
                    "system": TPL_PROMPT_CHAT_SYSTEM,
                    "user": TPL_PROMPT_CHAT_USER,
                },
                "word1_word2",
            ),
        ],
    )
    def test_chat_partial(self, tmp_prompt_dir: Path, tpls: dict, text: str | None):
        _write_chat_tpls(tmp_prompt_dir, tpls)

        registry = LocalPromptRegistry.from_resources(
            {
                "chat": tmp_prompt_dir,
            }
        )

        kwargs = {"text": text} if text else {}
        actual = registry.get_chat_prompt("chat", **kwargs)
        expected = _render_chat_tpl(tpls, **kwargs)

        assert actual == expected
