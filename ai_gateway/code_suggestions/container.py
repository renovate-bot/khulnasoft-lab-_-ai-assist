from dependency_injector import containers, providers

from ai_gateway.code_suggestions.completions import (
    CodeCompletions,
    CodeCompletionsLegacy,
)
from ai_gateway.code_suggestions.generations import CodeGenerations
from ai_gateway.code_suggestions.processing import ModelEngineCompletions
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor as PostProcessorCompletions,
)
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.experimentation import experiment_registry_provider
from ai_gateway.models import KindAnthropicModel, KindVertexTextModel, TextGenBaseModel
from ai_gateway.tokenizer import init_tokenizer

__all__ = [
    "ContainerCodeSuggestions",
]


class ContainerCodeGenerations(containers.DeclarativeContainer):
    tokenizer = providers.Dependency()
    vertex_code_bison = providers.Dependency(instance_of=TextGenBaseModel)
    anthropic_claude = providers.Dependency(instance_of=TextGenBaseModel)

    vertex = providers.Factory(
        CodeGenerations,
        model=providers.Factory(
            vertex_code_bison, name=KindVertexTextModel.CODE_BISON_002
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    # We need to resolve the model based on model name provided in request payload
    # Hence, CodeGenerations is only partially applied here.
    anthropic = providers.Factory(
        CodeGenerations,
        model=providers.Factory(
            providers.Factory(
                anthropic_claude,
            ),
            stop_sequences=["</new_code>", "\n\nHuman:"],
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )


class ContainerCodeCompletions(containers.DeclarativeContainer):
    tokenizer = providers.Dependency()
    vertex_code_gecko = providers.Dependency(instance_of=TextGenBaseModel)
    anthropic_claude = providers.Dependency(instance_of=TextGenBaseModel)

    config = providers.Configuration()

    vertex_legacy = providers.Factory(
        CodeCompletionsLegacy,
        engine=providers.Factory(
            ModelEngineCompletions,
            model=providers.Factory(
                vertex_code_gecko, name=KindVertexTextModel.CODE_GECKO_002
            ),
            tokenizer=tokenizer,
            experiment_registry=experiment_registry_provider(),
        ),
        post_processor=providers.Factory(
            PostProcessorCompletions,
            exclude=config.excl_post_proc,
        ).provider,
    )

    anthropic = providers.Factory(
        CodeCompletions,
        model=providers.Factory(
            providers.Factory(
                anthropic_claude, name=KindAnthropicModel.CLAUDE_INSTANT_1_2
            ),
            stop_sequences=["</new_code>", "\n\nHuman:"],
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )


class ContainerCodeSuggestions(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    config = providers.Configuration()

    tokenizer = providers.Resource(init_tokenizer)

    generations = providers.Container(
        ContainerCodeGenerations,
        tokenizer=tokenizer,
        vertex_code_bison=models.vertex_code_bison,
        anthropic_claude=models.anthropic_claude,
    )

    completions = providers.Container(
        ContainerCodeCompletions,
        tokenizer=tokenizer,
        vertex_code_gecko=models.vertex_code_gecko,
        anthropic_claude=models.anthropic_claude,
        config=config,
    )
