import os

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def init_tokenizer() -> PreTrainedTokenizerFast:
    # T5Tokenizer ignores new lines, tabs and multiple spaces used a lot in coding.
    # When the T5Tokenizer is applied, the output differs from input.
    # We're switching to the Salesforce Codegen tokenizer temporarily.
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B", use_fast=True)

    # Disable parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return tokenizer
