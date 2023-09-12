import pytest

from ai_gateway.prompts.parsers import CodeParser
from ai_gateway.suggestions.processing.base import LanguageId

GO_SOURCE_SAMPLE = """package main

import (
    "context"
    "flag"

    "gitlab.com/gitlab-org/labkit/fips"
)

func main() {

}

func myFunction(x int, y int) int {
  return x + y
}
"""

C_SOURCE_SAMPLE = """// C test

#include <stdio.h>

#define TEST 1

int main() {
  printf("Hello World!");
  return 0;
}

void myFunction(int myNumbers[5]) {
  for (int i = 0; i < 5; i++) {
    printf("%d\n", myNumbers[i]);
  }
}
"""

JAVA_SOURCE_SAMPLE = """// Java test
import java.util.ArrayList;

static void myMethod(String fname) {
    System.out.println(fname + " Refsnes");
}

public static void main(String[] args) {
    myMethod("Liam");
    myMethod("Jenny");
    myMethod("Anja");
}

"""

JAVASCRIPT_SOURCE_SAMPLE = """// Javascript test
import { someFunction } from './module';

// Function to compute the product of p1 and p2
function myFunction(p1, p2) {
  return p1 * p2;
}
"""

TYPESCRIPT_SOURCE_SAMPLE = """
let firstName = "Awesome";
function add(a: number, b: number, c?: number) {
  return a + b + (c || 0);
}
"""

PHP_SOURCE_SAMPLE = """<?php>
use SomeNamespace\\SomeClass;

function familyName($fname, $year) {
  echo "$fname Refsnes. Born in $year <br>";
}

familyName("Hege", "1975");
<?>
"""

PYTHON_SOURCE_SAMPLE = """# Python test
import os
def greet(name: str, age: int) -> None:
    print(f"Hello {name}, you are {age} years old!")

a = 1+1
"""

RUBY_SOURCE_SAMPLE = """# Ruby test
require 'date'
require_relative 'lib/test'

def test(a1 = "Ruby", a2 = "Perl")
   puts "The programming language is #{a1}"
   puts "The programming language is #{a2}"
end
test "C", "C++"
test
"""

RUST_SOURCE_SAMPLE = """// Rust test
use std::collections::HashMap;

fn init_map(items: Vec<String>, count: Vec<usize>) -> HashMap<String, usize> {
    let mut scores = HashMap::new();

    let zipped = items.iter().zip(count.iter());
    for (item, c) in zipped {
        scores.insert(String::from(item), c);
    }
    scores
}
"""

SCALA_SOURCE_SAMPLE = """// Scala test
import java.util._

def addInt( a:Int, b:Int ) : Int = {
    var sum:Int = 0
    sum = a + b
    return sum
}
"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "expected_outputs"),
    [
        (
            LanguageId.C,
            C_SOURCE_SAMPLE,
            ["int main()", "void myFunction(int myNumbers[5])"],
        ),
        (
            LanguageId.CPP,
            C_SOURCE_SAMPLE,
            ["int main()", "void myFunction(int myNumbers[5])"],
        ),
        (
            LanguageId.GO,
            GO_SOURCE_SAMPLE,
            ["func main()", "func myFunction(x int, y int) int"],
        ),
        (
            LanguageId.JS,
            JAVASCRIPT_SOURCE_SAMPLE,
            ["function myFunction(p1, p2)"],
        ),
        (
            LanguageId.TS,
            TYPESCRIPT_SOURCE_SAMPLE,
            ["function add(a: number, b: number, c?: number)"],
        ),
        (LanguageId.PHP, PHP_SOURCE_SAMPLE, ["function familyName($fname, $year)"]),
        (
            LanguageId.PYTHON,
            PYTHON_SOURCE_SAMPLE,
            ["def greet(name: str, age: int) -> None:"],
        ),
        (
            LanguageId.RUBY,
            RUBY_SOURCE_SAMPLE,
            ['def test(a1 = "Ruby", a2 = "Perl")'],
        ),
        (
            LanguageId.RUST,
            RUST_SOURCE_SAMPLE,
            [
                "fn init_map(items: Vec<String>, count: Vec<usize>) -> HashMap<String, usize>"
            ],
        ),
        (LanguageId.SCALA, SCALA_SOURCE_SAMPLE, ["def addInt( a:Int, b:Int ) : Int ="]),
    ],
)
def test_import_extractor(lang_id: LanguageId, source_code: str, expected_outputs: str):
    parser = CodeParser.from_language_id(source_code, lang_id)

    outputs = parser.function_signatures()

    assert len(outputs) == len(expected_outputs)
    for output, exp in zip(outputs, expected_outputs):
        assert string_equal(output, exp)


@pytest.mark.parametrize(
    ("lang_id", "source_code"),
    [
        (LanguageId.C, "there is nothing, but int main sounds nice"),
        (LanguageId.CPP, "there is nothing, but int main sounds nice"),
        (LanguageId.GO, "no functions here"),
        (LanguageId.JAVA, "no functions"),
        (LanguageId.JS, "no functions here"),
        (LanguageId.PHP, "<html></html>"),
        (LanguageId.PYTHON, "no functions"),
        (LanguageId.RUBY, "no functions"),
        (LanguageId.RUST, "no functions here"),
        (LanguageId.SCALA, "no functions here"),
        (LanguageId.TS, "no functions here"),
    ],
)
def test_unparseable(lang_id: LanguageId, source_code: str):
    parser = CodeParser.from_language_id(source_code, lang_id)
    output = parser.function_signatures()

    assert len(output) == 0


def string_equal(a: str, b: str):
    """
    Determine if two strings are equal, but ignoring any leading or trailing newlines.
    """
    a = a.strip()
    b = b.strip()
    return a == b
