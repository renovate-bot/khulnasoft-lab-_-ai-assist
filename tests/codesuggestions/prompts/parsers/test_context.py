import pytest
from tree_sitter import Node

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.prompts.parsers.context_extractors import BaseContextVisitor
from codesuggestions.prompts.parsers.treetraversal import tree_dfs
from codesuggestions.suggestions.processing.ops import (
    LanguageId,
    find_cursor_position,
    split_on_point,
)

PYTHON_PREFIX_SAMPLE = """
from abc import ABC
from abc import abstractmethod

from tree_sitter import Node


class BaseVisitor(ABC):
    _TARGET_SYMBOL = None

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_earlier(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node("""

PYTHON_SUFFIX_SAMPLE = """node)


class BaseCodeParser(ABC):
    @abstractmethod
    def count_symbols(self) -> dict:
        pass

    @abstractmethod
    def imports(self) -> list[str]:
        pass

"""

_PYTHON_PREFIX_EXPECTED_FUNCTION_DEFINITION_CONTEXT = """
def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(
"""

_PYTHON_PREFIX_EXPECTED_CLASS_DEFINITION_CONTEXT = """
class BaseVisitor(ABC):
    _TARGET_SYMBOL = None

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_earlier(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(
"""


@pytest.mark.parametrize(
    (
        "lang_id",
        "source_code",
        "target_point",
        "expected_context",
        "priority_list",
    ),
    [
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE,
            (21, 29),
            _PYTHON_PREFIX_EXPECTED_FUNCTION_DEFINITION_CONTEXT,
            ["function_definition"],
        ),
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE,
            (21, 29),
            _PYTHON_PREFIX_EXPECTED_CLASS_DEFINITION_CONTEXT,
            ["class_definition"],
        ),
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE + PYTHON_SUFFIX_SAMPLE,
            (21, 29),
            """def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(node)
""",
            ["function_definition"],
        ),
    ],
)
def test_base_context_visitor(
    lang_id: LanguageId,
    source_code: str,
    target_point: tuple[int, int],
    expected_context: str,
    priority_list: list[str],
):
    parser = CodeParser.from_language_id(source_code, lang_id)
    visitor = BaseContextVisitor(target_point)
    tree_dfs(parser.tree, visitor)

    context_node = visitor.extract_most_relevant_context(priority_list=priority_list)
    assert context_node is not None

    actual_context = visitor._bytes_to_str(context_node.text)
    assert actual_context.strip() == expected_context.strip()


PYTHON_SAMPLE_TWO_FUNCTIONS = """
def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

PYTHON_SAMPLE_FUNCTION_WITHIN_FUNCTION = """
import os

def i_want_to_sum(a, b):
    def sum(a, b):
        return a + b
    return sum(a, b)
"""

PYTHON_SAMPLE_TWO_CLASSES = """
from abc import ABC, abstractmethod

from tree_sitter import Node

__all__ = [
    "BaseVisitor",
    "BaseCodeParser",
]


class BaseVisitor(ABC):
    _TARGET_SYMBOLS = []

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_tree_traversal(self) -> bool:
        return False

    @property
    def stop_node_traversal(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)

    def _bytes_to_str(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")


class BaseCodeParser(ABC):
    @abstractmethod
    def count_symbols(self) -> dict:
        pass

    @abstractmethod
    def imports(self) -> list[str]:
        pass
"""

PYTHON_SAMPLE_CLASS_WITHIN_CLASS = """
class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""

JAVASCRIPT_SOURCE_SAMPLE = """
import React, { useState } from "react";
import dateFns from "date-fns";
import { sum } from "mathjs";

const App = () => {
  const [date, setDate] = useState(new Date());
  const [number, setNumber] = useState(0);

  const addNumber = () => {
    setNumber(sum(number, 1));
  };

  const getDateString = () => {
    return dateFns.format(date, "YYYY-MM-DD");
  };

  return (
    <div>
      <h1>Date: {getDateString()}</h1>
      <h1>Number: {number}</h1>
      <button onClick={addNumber}>Add 1</button>
    </div>
  );
};

export default App;
"""


JAVASCRIPT_TWO_CLASSES = """
class Animal {
  constructor(name, species) {
    this.name = name;
    this.species = species;
  }

  makeSound() {
    console.log(`${this.name} makes a sound`);
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name, 'Dog');
    this.breed = breed;
  }

  makeSound() {
    console.log(`${this.name} barks`);
  }

  fetch() {
    console.log(`${this.name} fetches the ball`);
  }
}

const animal = new Animal('Generic Animal', 'Unknown');
const dog = new Dog('Buddy', 'Golden Retriever');

animal.makeSound();
dog.makeSound();
dog.fetch();
"""

JAVASCRIPT_FUNCTION_SAMPLE = """
function dangerous_eval(myeval) {
    let x = 10;
    eval(myeval);
}

# more stuff that we don't care
function dangerous_eval_string(myeval) {
    eval('var myeval = "' + myeval + '";');
"""

JAVASCRIPT_GENERATOR_FUNCTION_SAMPLE = """
function* animalSounds(name, species) {
  yield `${name} the ${species} makes a sound`;
}

function* dogSounds(name, breed) {
  yield* animalSounds(name, 'Dog');
  yield `${name} the ${breed} barks`;
}

function* dogActions(name, breed) {
  yield* dogSounds(name, breed);
  yield `${name} the ${breed} fetches the ball`;
}
"""

JAVASCRIPT_LEXICAL_WITH_GENERATOR_SAMPLE = """
const generateSounds = (name, species) => function* () {
  yield `${name} the ${species} makes a sound`;
};

const generateDogSounds = (name, breed) => function* () {
  yield* generateSounds(name, 'Dog')();
  yield `${name} the ${breed} barks`;
};

const generateDogActions = (name, breed) => function* () {
  yield* generateDogSounds(name, breed)();
  yield `${name} the ${breed} fetches the ball`;
};

const animalSoundGenerator = generateSounds('Generic Animal', 'Unknown');
const dogSoundGenerator = generateDogSounds('Buddy', 'Golden Retriever');
const dogActionGenerator = generateDogActions('Buddy', 'Golden Retriever');

for (const sound of animalSoundGenerator()) {
  console.log(sound);
}
"""

JAVASCRIPT_FUNCTION_WITHIN_FUNCTION = """
function outerFunction() {
  console.log("This is the outer function.");

  function innerFunction() {
    console.log("This is the inner function.");
  }

  function* generatorFunction() {
    yield `hello govna`;
  }

  innerFunction(); // Call the inner function
}

outerFunction(); // Call the outer function
"""

TYPESCRIPT_INTERFACE_SAMPLE = """
interface Person {
  firstName: string;
  lastName: string;
  age: number;
  sayHello: () => void;
}

class Student implements Person {
  firstName: string;
  lastName: string;
  age: number;

  constructor(firstName: string, lastName: string, age: number) {
    this.firstName = firstName;
    this.lastName = lastName;
    this.age = age;
  }

  sayHello() {
    console.log(`Hello, govna`);
  }
}
"""

TYPESCRIPT_CALL_EXPRESSION_SAMPLE = """
import "@testing-library/cypress/add-commands";
import "./commands";

Cypress.on("uncaught:exception", (err) => {
  return false;
});

Cypress.on("before:run", () => { // don't care
"""


@pytest.mark.parametrize(
    (
        "lang_id",
        "source_code",
        "target_point",
        "expected_prefix",
        "expected_suffix",
    ),
    [
        # TODO: Add a test to sweep the full range of the context rectangle
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 14),
            "def sum(a, b):",
            "\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 13),
            "def sum(a, b)",
            ":\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 12),
            "def sum(a, b",
            "):\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 11),
            "def sum(a, ",
            "b):\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_FUNCTION_WITHIN_FUNCTION[1:],
            (2, 20),
            "import os\n\ndef i_want_to_sum(a,",
            " b):\n    def sum(a, b):\n        return a + b\n    return sum(a, b)",
        ),
        (  # Test context at module level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (2, 0),
            "def sum(a, b):\n    return a + b\n",
            "\ndef subtract(a, b):\n    return a - b\n",
        ),
        (  # Test context at class level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_CLASSES[1:],
            (25, 32),
            # fmt: off
            PYTHON_SAMPLE_TWO_CLASSES[1:438], # last line:'     def visit(self, node: Node):'
"""
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)

    def _bytes_to_str(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within nested the class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (4, 21),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:130],
""": str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within the 2nd nested class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (7, 15),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:191],
""" str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within outer class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (11, 0),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:233],
"""
    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[1:-1],
            # fmt: on
        ),
        (  # Test context at lexical declaration level
            LanguageId.JS,
            JAVASCRIPT_SOURCE_SAMPLE[1:],
            (9, 0),
            JAVASCRIPT_SOURCE_SAMPLE[1:245],
            """
    setNumber(sum(number, 1));
  };

  const getDateString = () => {
    return dateFns.format(date, "YYYY-MM-DD");
  };

  return (
    <div>
      <h1>Date: {getDateString()}</h1>
      <h1>Number: {number}</h1>
      <button onClick={addNumber}>Add 1</button>
    </div>
  );
};
"""[
                1:-1
            ],
        ),
        (  # Test context class level
            LanguageId.JS,
            JAVASCRIPT_TWO_CLASSES[1:],
            (3, 16),
            JAVASCRIPT_TWO_CLASSES[1:85],
            # fmt: off
            """
 = species;
  }

  makeSound() {
    console.log(`${this.name} makes a sound`);
  }
}
"""[1:-1]
            # fmt: on
        ),
        (  # Test context function level
            LanguageId.JS,
            # fmt: off
            JAVASCRIPT_FUNCTION_SAMPLE[1:],
            (2, 0),
            JAVASCRIPT_FUNCTION_SAMPLE[1:51],
            "    eval(myeval);\n}"
            # fmt: on
        ),
        (  # Test context generator function level
            LanguageId.JS,
            # fmt: off
            JAVASCRIPT_GENERATOR_FUNCTION_SAMPLE[1:],
            (5, 0),
            JAVASCRIPT_GENERATOR_FUNCTION_SAMPLE[1:127],
            """
  yield* animalSounds(name, 'Dog');
  yield `${name} the ${breed} barks`;
}
"""[1:-1]
            # fmt: on
        ),
        (  # Test context generator function within arrow function
            LanguageId.JS,
            # fmt: off
            JAVASCRIPT_LEXICAL_WITH_GENERATOR_SAMPLE[1:],
            (6, 0),
            JAVASCRIPT_LEXICAL_WITH_GENERATOR_SAMPLE[1:208],
            """
  yield `${name} the ${breed} barks`;
};
"""[1:-1]
            # fmt: on
        ),
        (  # Test context function within function, cursor within inner function
            LanguageId.JS,
            # fmt: off
            JAVASCRIPT_FUNCTION_WITHIN_FUNCTION[1:],
            (4, 0),
            JAVASCRIPT_FUNCTION_WITHIN_FUNCTION[1:104],
            """
    console.log("This is the inner function.");
  }

  function* generatorFunction() {
    yield `hello govna`;
  }

  innerFunction(); // Call the inner function
}
"""[1:-1]
            # fmt: on
        ),
        (  # Test context function within function, cursor within a generator function
            LanguageId.JS,
            # fmt: off
            JAVASCRIPT_FUNCTION_WITHIN_FUNCTION[1:],
            (8, 0),
            JAVASCRIPT_FUNCTION_WITHIN_FUNCTION[1:191],
            """
    yield `hello govna`;
  }

  innerFunction(); // Call the inner function
}
"""[1:-1],
        ),
        (  # TS: Test interface
            LanguageId.TS,
            TYPESCRIPT_INTERFACE_SAMPLE[1:],
            (3, 0),
            # fmt: off
            TYPESCRIPT_INTERFACE_SAMPLE[1:61],
"""
  age: number;
  sayHello: () => void;
}
"""[1:-1],
            # fmt: on
        ),
        (  # TS: Test interface
            LanguageId.TS,
            TYPESCRIPT_CALL_EXPRESSION_SAMPLE[1:],
            (4, 0),
            # fmt: off
            TYPESCRIPT_CALL_EXPRESSION_SAMPLE[1:115],
"""
  return false;
})
"""[1:-1],
            # fmt: on
        ),
    ],
)
def test_suffix_near_cursor(
    lang_id: LanguageId,
    source_code: str,
    target_point: tuple[int, int],
    expected_prefix: str,
    expected_suffix: str,
):
    parser = CodeParser.from_language_id(source_code, lang_id)
    actual_prefix, _ = split_on_point(source_code, target_point)

    print(f"{target_point=}")
    print("-----------------------")
    print("source_code:")
    print("-----------------------")
    pos = find_cursor_position(source_code, target_point)
    print(_highlight_position(pos, source_code))

    actual_truncated_suffix = parser.suffix_near_cursor(target_point)

    print("-----------------------")
    print("Prefix")
    print("-----------------------")
    print(repr(actual_prefix))
    print(repr(expected_prefix))

    print("-----------------------")
    print("Suffix")
    print("-----------------------")
    print(repr(actual_truncated_suffix))
    print(repr(expected_suffix))

    assert actual_prefix == expected_prefix
    assert actual_truncated_suffix == expected_suffix


def _highlight_position(pos, mystring):
    # fix this quadratic loop
    text_highlight = ""
    for i, x in enumerate(mystring):
        if i == pos:
            text_highlight += f"\033[44;33m{x}\033[m"
        else:
            text_highlight += x
    return text_highlight
