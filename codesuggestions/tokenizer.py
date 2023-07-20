from transformers import AutoTokenizer


def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B", use_fast=True)

    return tokenizer
