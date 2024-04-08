from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from jinja2 import (
    Environment,
    FileSystemLoader,
    PrefixLoader,
    Template,
    TemplateNotFound,
)

from ai_gateway.chat.prompts import BasePromptRegistry, ChatPrompt

__all__ = ["LocalPromptRegistry"]


def list_to_tuple(fn: Callable) -> Callable:
    """
    `lru_cache` requires the passed parameters to be hashable
    """

    def wrapper(*args: Any, **kwargs: Any):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        kwargs = {k: tuple(x) if isinstance(x, list) else x for k, x in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


class LocalPromptRegistry(BasePromptRegistry):
    def __init__(self, env_jinja: Environment):
        self.env_jinja = env_jinja

    def _get_template(
        self, name: str, ignore_exception: bool = False
    ) -> Optional[Template]:
        tpl = None

        try:
            tpl = self.env_jinja.get_template(name)
        except TemplateNotFound as ex:
            if not ignore_exception:
                raise ex

        return tpl

    @list_to_tuple
    @lru_cache
    def get_prompt(self, key: str, **kwargs: Any) -> str:
        template = self._get_template(f"{key}/tpl.jinja")

        return template.render(**kwargs)

    @list_to_tuple
    @lru_cache
    def get_chat_prompt(self, key: str, **kwargs: Any) -> ChatPrompt:
        user = self._get_template(f"{key}/user.jinja")
        system = self._get_template(f"{key}/system.jinja", ignore_exception=True)
        assistant = self._get_template(f"{key}/assistant.jinja", ignore_exception=True)

        return ChatPrompt(
            user=user.render(**kwargs),
            system=system.render(**kwargs) if system else None,
            assistant=assistant.render(**kwargs) if assistant else None,
        )

    @classmethod
    def from_resources(cls, mapping: dict[str, Path]) -> "LocalPromptRegistry":
        env_jinja = Environment(
            loader=PrefixLoader(
                {key: FileSystemLoader(path) for key, path in mapping.items()}
            )
        )

        return cls(env_jinja)
