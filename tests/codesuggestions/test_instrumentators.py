from unittest import mock

import pytest
from starlette_context import request_cycle_context

from ai_gateway.instrumentators.base import TextGenModelInstrumentator
from ai_gateway.suggestions.processing import (
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
)
from ai_gateway.suggestions.processing.completions import Prompt


class TestTextGenModelInstrumentator:
    @mock.patch("prometheus_client.Counter.labels")
    def test_cost_metric_counts_stripped_model_input_output(self, mock_labels):
        prefix = "a b c"  # expected len: 3
        suffix = "d\ne"  # expected len: 2
        metadata = MetadataPromptBuilder(
            components={
                "prefix": MetadataCodeContent(length=10, length_tokens=2),
                "suffix": MetadataCodeContent(length=10, length_tokens=2),
            },
            imports=MetadataExtraInfo(
                name="name",
                pre=MetadataCodeContent(length=0, length_tokens=0),
                post=MetadataCodeContent(length=0, length_tokens=0),
            ),
        )
        prompt = Prompt(prefix=prefix, suffix=suffix, metadata=metadata)
        model_engine = "vertex-ai"
        model_name = "code-gecko"
        feature_category = "code_suggestions"
        completion = "e f g"  # expected len: 3

        context = {}

        instrumentator = TextGenModelInstrumentator(
            model_engine=model_engine, model_name=model_name
        )

        with request_cycle_context(context):
            with instrumentator.watch(
                prompt, suffix_length=len(suffix)
            ) as watch_container:
                watch_container.register_model_output_length(completion)

        mock_labels.assert_has_calls(
            [
                # track inference count
                mock.call(model_engine="vertex-ai", model_name="code-gecko"),
                mock.call().inc(),
                # track model cost input
                mock.call(
                    item="completions/completion/input",
                    unit="characters",
                    vendor="vertex-ai",
                    model="code-gecko",
                    feature_category="code_suggestions",
                ),
                mock.call().inc(5),  # prefix + suffix
                # track model cost output
                mock.call(
                    item="completions/completion/output",
                    unit="characters",
                    vendor="vertex-ai",
                    model="code-gecko",
                    feature_category="code_suggestions",
                ),
                mock.call().inc(3),
            ]
        )
