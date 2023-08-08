import pytest

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.suggestions.processing.base import LanguageId

GO_SOURCE_SAMPLE = """package main

import (
    "context"
    "flag"

    "gitlab.com/gitlab-org/labkit/fips"
)

func main() {

}
"""

C_SOURCE_SAMPLE = """// C test

#include <stdio.h>

#define TEST 1
"""

JAVA_SOURCE_SAMPLE = """// Java test
import java.util.ArrayList;
import static java.lang.Math.sqrt;
"""

JAVASCRIPT_SOURCE_SAMPLE = """// Javascript test
import { someFunction } from './module';
"""

PHP_SOURCE_SAMPLE = """<?php>
use SomeNamespace\\SomeClass;
<?>
"""

PYTHON_SOURCE_SAMPLE = """# Python test
import os
from abc import ABC
"""

RUBY_SOURCE_SAMPLE = """# Ruby test
require 'date'
require_relative 'lib/test'
"""

RUST_SOURCE_SAMPLE = """// Rust test
use std::collections::HashMap;
"""

SCALA_SOURCE_SAMPLE = """// Scala test
import java.util._
"""

KOTLIN_SOURCE_SAMPLE = """// Kotlin test
import java.util.Random
"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "expected_output"),
    [
        (LanguageId.C, C_SOURCE_SAMPLE, ["#include <stdio.h>\n"]),
        (LanguageId.CPP, C_SOURCE_SAMPLE, ["#include <stdio.h>\n"]),
        (
            LanguageId.GO,
            GO_SOURCE_SAMPLE,
            ["\n".join(GO_SOURCE_SAMPLE.split("\n")[2:8])],
        ),
        (
            LanguageId.JAVA,
            JAVA_SOURCE_SAMPLE,
            ["import java.util.ArrayList;", "import static java.lang.Math.sqrt;"],
        ),
        (
            LanguageId.JS,
            JAVASCRIPT_SOURCE_SAMPLE,
            ["import { someFunction } from './module';"],
        ),
        (LanguageId.PHP, PHP_SOURCE_SAMPLE, ["use SomeNamespace\\SomeClass;"]),
        (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE, ["import os", "from abc import ABC"]),
        (
            LanguageId.RUBY,
            RUBY_SOURCE_SAMPLE,
            ["require 'date'", "require_relative 'lib/test'"],
        ),
        (LanguageId.RUST, RUST_SOURCE_SAMPLE, ["use std::collections::HashMap;"]),
        (LanguageId.SCALA, SCALA_SOURCE_SAMPLE, ["import java.util._"]),
        (
            LanguageId.TS,
            JAVASCRIPT_SOURCE_SAMPLE,
            ["import { someFunction } from './module';"],
        ),
        (LanguageId.KOTLIN, KOTLIN_SOURCE_SAMPLE, ["import java.util.Random"]),
    ],
)
def test_import_extractor(lang_id: LanguageId, source_code: str, expected_output: str):
    parser = CodeParser.from_language_id(source_code, lang_id)

    output = parser.imports()

    assert output == expected_output


@pytest.mark.parametrize(
    ("lang_id", "source_code"),
    [
        (LanguageId.C, "there is nothing, but #include sounds nice"),
        (LanguageId.CPP, "there is nothing, but #include sounds nice"),
        (LanguageId.GO, "nothing to import here"),
        (LanguageId.JAVA, "nothing to import"),
        (LanguageId.JS, "nothing to import here"),
        (LanguageId.PHP, "<html></html>"),
        (LanguageId.PYTHON, "nothing to import"),
        (LanguageId.RUBY, "nothing to import"),
        (LanguageId.RUST, "nothing to use here"),
        (LanguageId.SCALA, "nothing to import here"),
        (LanguageId.TS, "nothing to import here"),
        (LanguageId.KOTLIN, "nothing to import here"),
    ],
)
def test_unparseable(lang_id: LanguageId, source_code: str):
    parser = CodeParser.from_language_id(source_code, lang_id)
    output = parser.imports()

    assert len(output) == 0
