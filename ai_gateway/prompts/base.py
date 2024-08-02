from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple, TypeVar, cast

from jinja2 import BaseLoader, Environment
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import Runnable, RunnableBinding

from ai_gateway.auth.user import GitLabUser
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.typing import ModelMetadata, TypeModelFactory

__all__ = [
    "Prompt",
    "BasePromptRegistry",
]

Input = TypeVar("Input")
Output = TypeVar("Output")

jinja_env = Environment(loader=BaseLoader())


def _format_str(content: str, options: dict[str, Any]) -> str:
    return jinja_env.from_string(content).render(options)


class Prompt(RunnableBinding[Input, Output]):
    name: str
    unit_primitives: list[GitLabUnitPrimitive]

    def __init__(
        self,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[ModelMetadata] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        model = self._build_model(
            model_factory, config.model, config.params, model_metadata
        )
        messages = self.build_messages(config.prompt_template, options or {})
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = self._build_chain(cast(Runnable[Input, Output], prompt | model))

        super().__init__(name=config.name, unit_primitives=config.unit_primitives, bound=chain)  # type: ignore[call-arg]

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
        params: PromptParams | None,
        model_metadata: Optional[ModelMetadata] | None,
    ) -> Runnable:
        model = model_factory(
            model=config.name,
            **config.params.model_dump(
                exclude={"model_class_provider"}, exclude_none=True, by_alias=True
            )
        )
        kwargs = {}

        if params:
            kwargs.update(params.model_dump())

        if model_metadata:
            kwargs.update(
                model=model_metadata.name,
                api_base=str(model_metadata.endpoint),
                custom_llm_provider=model_metadata.provider,
                api_key=model_metadata.api_key,
            )

        return model.bind(**kwargs)

    # Subclasses can override this method to add steps at either side of the chain
    @staticmethod
    def _build_chain(chain: Runnable[Input, Output]) -> Runnable[Input, Output]:
        return chain

    # Assume that the prompt template keys map to roles. Subclasses can
    # override this method to implement more complex logic.
    @staticmethod
    def _prompt_template_to_messages(
        tpl: dict[str, str], options: dict[str, Any]
    ) -> list[Tuple[str, str]]:
        return list(tpl.items())

    @classmethod
    def build_messages(
        cls, prompt_template: dict[str, str], options: dict[str, Any]
    ) -> Sequence[MessageLikeRepresentation]:
        messages = []

        for role, template in cls._prompt_template_to_messages(
            prompt_template, options
        ):
            messages.append((role, _format_str(template, options)))

        return messages


class BasePromptRegistry(ABC):
    @abstractmethod
    def get(
        self,
        prompt_id: str,
        options: Optional[dict[str, Any]] = None,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        pass

    def get_on_behalf(
        self,
        user: GitLabUser,
        prompt_id: str,
        options: Optional[dict[str, Any]] = None,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        prompt = self.get(prompt_id, options, model_metadata)

        for unit_primitive in prompt.unit_primitives:
            if not user.can(unit_primitive):
                raise WrongUnitPrimitives

        return prompt
