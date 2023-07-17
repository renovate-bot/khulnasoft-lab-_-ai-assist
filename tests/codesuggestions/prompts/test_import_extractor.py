from codesuggestions.prompts.import_extractor import ImportExtractor
from codesuggestions.suggestions.processing.base import LanguageId

import pytest


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
"""

RUST_SOURCE_SAMPLE = """// Rust test
use std::collections::HashMap;
"""

SCALA_SOURCE_SAMPLE = """// Scala test
import java.util._
"""


@pytest.mark.parametrize(("lang_id", "source_code", "expected_output"), [
    (LanguageId.C, C_SOURCE_SAMPLE, "#include <stdio.h>\n\n"),
    (LanguageId.CPP, C_SOURCE_SAMPLE, "#include <stdio.h>\n\n"),
    (LanguageId.GO, GO_SOURCE_SAMPLE, "\n".join(GO_SOURCE_SAMPLE.split("\n")[2:8])),
    (LanguageId.JAVA, JAVA_SOURCE_SAMPLE, "import java.util.ArrayList;"),
    (LanguageId.JS, JAVASCRIPT_SOURCE_SAMPLE, "import { someFunction } from './module';"),
    (LanguageId.PHP, PHP_SOURCE_SAMPLE, "use SomeNamespace\\SomeClass;"),
    (LanguageId.PYTHON, PYTHON_SOURCE_SAMPLE, "import os"),
    (LanguageId.RUST, RUST_SOURCE_SAMPLE, "use std::collections::HashMap;"),
    (LanguageId.SCALA, SCALA_SOURCE_SAMPLE, "import java.util._"),
    (LanguageId.TS, JAVASCRIPT_SOURCE_SAMPLE, "import { someFunction } from './module';")
])
def test_import_extractor(lang_id: LanguageId, source_code: str, expected_output: str):
    extractor = ImportExtractor(lang_id)
    output = extractor.extract_imports(source_code)

    assert len(output) == 1
    assert output == [expected_output]


@pytest.mark.parametrize(("lang_id", "source_code"), [
    (LanguageId.C, "there is nothing, but #include sounds nice"),
    (LanguageId.CPP, "there is nothing, but #include sounds nice"),
    (LanguageId.GO, "nothing to import here"),
    (LanguageId.JAVA, "nothing to import"),
    (LanguageId.JS, "nothing to import here"),
    (LanguageId.PHP, "<html></html>"),
    (LanguageId.PYTHON, "nothing to import"),
    (LanguageId.RUST, "nothing to use here"),
    (LanguageId.SCALA, "nothing to import here"),
    (LanguageId.TS, "nothing to import here"),
])
def test_unparseable(lang_id: LanguageId, source_code: str):
    extractor = ImportExtractor(lang_id)
    output = extractor.extract_imports(source_code)

    assert len(output) == 0


@pytest.mark.parametrize(("lang_id"), [
   (LanguageId.KOTLIN),
])
def test_unsupported_languages(lang_id: LanguageId):
    extractor = ImportExtractor(lang_id)

    assert extractor.extract_imports("import java.util.*") == []


def test_non_utf8():
    value = b'\xc3\x28'  # Invalid UTF-8 byte sequence

    extractor = ImportExtractor(LanguageId.JS)
    assert extractor.extract_imports(value) == []
