from typing import Union

import pytest
from transformers import AutoTokenizer

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataPromptBuilder,
    Prompt,
)
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenizerTokenStrategy,
)
from ai_gateway.prompts import PromptTemplate

# This template takes 4 tokens (ignore placeholders)
# with the Salesforce/codegen2-16B tokenizer
_TEST_TEMPLATE_1 = """
start
{prefix}
end
""".strip(
    "\n"
)

# This template takes 4 tokens (ignore placeholders)
# with the Salesforce/codegen2-16B tokenizer
_TEST_TEMPLATE_2 = """
start {lang}
{prefix}
end
""".strip(
    "\n"
)


class TestPromptBuilderPrefixBased:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")

    @pytest.mark.parametrize(
        ("prefix", "total_max_len", "expected_prompt"),
        [
            (
                "random_text",
                2048,
                Prompt(
                    prefix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                1,
                Prompt(
                    prefix="text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
            (
                ["random_text", "random_another_text"],
                4,
                Prompt(
                    prefix="random_text\ntext",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=16, length_tokens=4)
                        }
                    ),
                ),
            ),
        ],
    )
    def test_without_template(
        self, prefix: Union[str, list[str]], total_max_len: int, expected_prompt: Prompt
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(prefix)
        else:
            builder.add_content(*prefix)

        actual = builder.build()

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        ("prefix", "total_max_len", "tpl", "tpl_args", "expected_prompt"),
        [
            (
                "random_text",
                2048,
                _TEST_TEMPLATE_1,
                dict(),
                Prompt(
                    prefix="start\nrandom_text\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                5,
                _TEST_TEMPLATE_1,
                dict(),
                Prompt(
                    prefix="start\ntext\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                2048,
                _TEST_TEMPLATE_2,
                dict(lang="python"),
                Prompt(
                    prefix="start python\nrandom_text\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                6,
                _TEST_TEMPLATE_2,
                dict(lang="python"),
                Prompt(
                    prefix="start python\ntext\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
        ],
    )
    def test_with_template(
        self,
        prefix: Union[str, list[str]],
        total_max_len: int,
        tpl: str,
        tpl_args: dict,
        expected_prompt: Prompt,
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(prefix)
        else:
            builder.add_content(*prefix)

        builder.add_template(PromptTemplate(tpl), **tpl_args)
        actual = builder.build()

        assert actual == expected_prompt
