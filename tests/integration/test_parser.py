import time

from ai_gateway.code_suggestions.processing.base import LanguageId
from ai_gateway.prompts.parsers import CodeParser

SOURCE_CODE = """
import os
from abc import ABC

{comments}

import regex
"""


def test_benchmark_parser():
    for i in range(90_000, 200_000, 10_000):
        source_code = SOURCE_CODE.format(
            comments="\n".join(["// A random comment"] * i)
        )
        lang_id = LanguageId.PYTHON

        parser = CodeParser.from_language_id(source_code, lang_id)

        start_time = time.perf_counter()
        output = parser.imports()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"{i}:{elapsed_time}", end="\n")


test_benchmark_parser()
