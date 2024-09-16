from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Mapping, Optional, Sequence, Tuple, TypeVar, cast

from jinja2 import PackageLoader
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig

from ai_gateway.auth.user import GitLabUser
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.typing import ModelMetadata, TypeModelFactory

__all__ = [
    "Prompt",
    "BasePromptRegistry",
]

Input = TypeVar("Input")
Output = TypeVar("Output")

jinja_env = SandboxedEnvironment(
    loader=PackageLoader("ai_gateway.prompts", "definitions")
)


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    return jinja_env.from_string(template).render(**kwargs)


# Override LangChain's jinja2 formatter so we can specify a loader with access to all our templates
DEFAULT_FORMATTER_MAPPING["jinja2"] = jinja2_formatter


class Prompt(RunnableBinding[Input, Output]):
    name: str
    model: BaseChatModel
    unit_primitives: list[GitLabUnitPrimitive]

    def __init__(
        self,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[ModelMetadata] = None,
        **kwargs,
    ):
        model_kwargs = self._build_model_kwargs(config.params, model_metadata)
        model = self._build_model(model_factory, config.model)
        messages = self.build_messages(config.prompt_template, **kwargs)
        prompt = ChatPromptTemplate.from_messages(messages, template_format="jinja2")
        chain = self._build_chain(
            cast(Runnable[Input, Output], prompt | model.bind(**model_kwargs))
        )

        super().__init__(name=config.name, model=model, unit_primitives=config.unit_primitives, bound=chain)  # type: ignore[call-arg]

    def _build_model_kwargs(
        self,
        params: PromptParams | None,
        model_metadata: Optional[ModelMetadata] | None,
    ) -> Mapping[str, Any]:
        kwargs = {}

        if params:
            kwargs.update(params.model_dump(exclude_none=True))

        if model_metadata:
            kwargs.update(
                model=model_metadata.name,
                api_base=str(model_metadata.endpoint),
                custom_llm_provider=model_metadata.provider,
                api_key=model_metadata.api_key,
            )

        return kwargs

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
    ) -> BaseChatModel:
        return model_factory(
            model=config.name,
            **config.params.model_dump(
                exclude={"model_class_provider"}, exclude_none=True, by_alias=True
            ),
        )

    @property
    def model_name(self) -> str:
        return self.model._identifying_params["model"]

    @property
    def instrumentator(self) -> ModelRequestInstrumentator:
        return ModelRequestInstrumentator(
            model_engine=self.model._llm_type,
            model_name=self.model_name,
            concurrency_limit=None,  # TODO: Plug concurrency limit into agents
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        with self.instrumentator.watch(stream=False):
            return await super().ainvoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        with self.instrumentator.watch(stream=True) as watcher:
            async for item in super().astream(input, config, **kwargs):
                yield item

            await watcher.afinish()

    # Subclasses can override this method to add steps at either side of the chain
    @staticmethod
    def _build_chain(chain: Runnable[Input, Output]) -> Runnable[Input, Output]:
        return chain

    # Assume that the prompt template keys map to roles. Subclasses can
    # override this method to implement more complex logic.
    @staticmethod
    def _prompt_template_to_messages(tpl: dict[str, str]) -> list[Tuple[str, str]]:
        return list(tpl.items())

    @classmethod
    def build_messages(
        cls, prompt_template: dict[str, str], **kwargs
    ) -> Sequence[MessageLikeRepresentation]:
        messages = []

        for role, template in cls._prompt_template_to_messages(prompt_template):
            messages.append((role, template))

        return messages


class BasePromptRegistry(ABC):
    @abstractmethod
    def get(
        self,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        pass

    def get_on_behalf(
        self,
        user: GitLabUser,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        prompt = self.get(prompt_id, model_metadata)

        for unit_primitive in prompt.unit_primitives:
            if not user.can(unit_primitive):
                raise WrongUnitPrimitives

        return prompt
