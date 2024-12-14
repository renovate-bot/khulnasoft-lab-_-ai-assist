from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Mapping, Optional, Tuple, TypeVar, cast

from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from jinja2 import PackageLoader
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.typing import Model, ModelMetadata, TypeModelFactory

__all__ = [
    "Prompt",
    "BasePromptRegistry",
    "jinja2_formatter",
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
    model: Model
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_tpl: Runnable[Input, PromptValue]

    def __init__(
        self,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[ModelMetadata] = None,
        disable_streaming: bool = False,
    ):
        model_kwargs = self._build_model_kwargs(config.params, model_metadata)
        model = self._build_model(model_factory, config.model, disable_streaming)
        prompt = self._build_prompt_template(config.prompt_template)
        chain = self._build_chain(
            cast(Runnable[Input, Output], prompt | model.bind(**model_kwargs))
        )

        super().__init__(
            name=config.name,
            model=model,
            unit_primitives=config.unit_primitives,
            bound=chain,
            prompt_tpl=prompt,
        )  # type: ignore[call-arg]

    def _build_model_kwargs(
        self,
        params: PromptParams | None,
        model_metadata: Optional[ModelMetadata] | None,
    ) -> Mapping[str, Any]:
        return {
            **(params.model_dump(exclude_none=True) if params else {}),
            **(model_metadata_to_params(model_metadata) if model_metadata else {}),
        }

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
        disable_streaming: bool,
    ) -> Model:
        return model_factory(
            model=config.name,
            disable_streaming=disable_streaming,
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
    def _build_prompt_template(
        cls, prompt_template: dict[str, str]
    ) -> Runnable[Input, PromptValue]:
        messages = []

        for role, template in cls._prompt_template_to_messages(prompt_template):
            messages.append((role, template))

        return cast(
            Runnable[Input, PromptValue],
            ChatPromptTemplate.from_messages(messages, template_format="jinja2"),
        )


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
        user: StarletteUser,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        prompt = self.get(prompt_id, model_metadata)

        for unit_primitive in prompt.unit_primitives:
            if not user.can(unit_primitive):
                raise WrongUnitPrimitives

        return prompt


def model_metadata_to_params(model_metadata: ModelMetadata) -> dict[str, str]:
    params = {
        "api_base": str(model_metadata.endpoint).removesuffix("/"),
        "api_key": str(model_metadata.api_key),
        "model": model_metadata.name,
        "custom_llm_provider": model_metadata.provider,
    }

    if not model_metadata.identifier:
        return params

    provider, _, model_name = model_metadata.identifier.partition("/")

    if model_name:
        params["custom_llm_provider"] = provider
        params["model"] = model_name

        if provider == "bedrock":
            del params["api_base"]
    else:
        params["custom_llm_provider"] = "custom_openai"
        params["model"] = model_metadata.identifier

    return params
