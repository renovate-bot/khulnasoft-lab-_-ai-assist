import pytest

from ai_gateway.code_suggestions.processing.post.ops import clean_model_reflection

PREFIX_JAVASCRIPT_1 = """
// This code has a filename of test-2.js and is written in JavaScript.
testing
""".strip(
    "\n"
)

COMPLETION_JAVASCRIPT_1_1 = """
only
// This code has a filename of test-2.js and is written in JavaScript.
testingonly
""".strip(
    "\n"
)

COMPLETION_JAVASCRIPT_1_2 = """
only
""".strip(
    "\n"
)

COMPLETION_JAVASCRIPT_1_3 = """
only

const another_function = () => {} 
""".strip(
    "\n"
)

PREFIX_JAVASCRIPT_2 = """
// this will be a function to read an address

const readAddress = (address) => {
    return new Promise((resolve, reject) => {
        loadFile(address)
       .then((data) => {
            resolve(data);
       })
      .catch ((err) => {
          reject(err);
      });
    });
};
""".lstrip(
    "\n"
)

COMPLETION_JAVASCRIPT_2_1 = """


// this will be a function to read an address

const readAddress = (address) => {
"""

PREFIX_JAVASCRIPT_3 = """
 // This code has a filename of test.js and is written in JavaScript.
import testing;

const newFunctionForValidatingEmail = (email) => {
  return emailRegex.test(email);
}

// For the mask XYZ
const writeStringBackwards = (inpStr) => {
  let outStr = "";
  for (let i = inpStr.length - 1; i >= 0; i--) {
    outStr += inpStr[i];
  }
  return outStr;
}

export {
"""

COMPLETION_JAVASCRIPT_3_1 = """
 newFunctionForValidatingEmail, writeStringBackwards };

// This code has a filename of test.js and is written in JavaScript.
import testing;
""".strip(
    "\n"
)

PREFIX_JAVASCRIPT_4 = """
 // This code has a filename of test.js and is written in JavaScript.
import testing;

const newFunctionForValidatingEmail = (email) => {
  return emailRegex.test(email);
}

// For the mask XYZ
const writeStringBackwards = (inpStr) => {
  let outStr = '';
  for (let i = inpStr.length - 1; i >= 0; i--) {
    outStr += inpStr[i];
  }
  return outSt
""".strip(
    "\n"
)

COMPLETION_JAVASCRIPT_4_1 = """
r;\n}\n\nconst maskXYZ = (inpStr) => {\n  let outStr = '';
""".strip(
    "\n"
)


PREFIX_PYTHON_1 = """
# This code has a filename of app.py and is written in Python.
def print_hello_world():
    print("Hello World!")
\t
""".strip(
    "\n"
)

COMPLETION_PYTHON_1_1 = """
# This code has a filename of app.py and is written in Python.
def print_hello_world():
    print("Hello World!")
"""

PREFIX_PYTHON_2 = """
# This code has a filename of app.py and is written in Python.
def print_hello_world():
\t
""".strip(
    "\n"
)

COMPLETION_PYTHON_2_1 = """
# This code has a filename of app.py and is written in Python.
\tprint("hello world")
""".strip(
    "\n"
)


PREFIX_RUBY_1 = """
# frozen_string_literal: true                    
                                                 
require 'carrierwave/orm/activerecord'           
                                                 
class Project < ApplicationRecord
  include Gitlab::ConfigHelper
  include Gitlab::VisibilityLevel

  def print_name
    puts name
  end

# Create new code for the following description: method which checks if name is valid
"""

COMPLETION_RUBY_1_1 = """
def valid_name?
  name.present? && name.length <= 255
end

# Create new code for the following description: method which checks if name is valid
"""


@pytest.mark.parametrize(
    ("context", "completion", "min_block_size", "expected"),
    [
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_1, 2, "only\n"),
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_2, 2, "only"),
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_3, 2, COMPLETION_JAVASCRIPT_1_3),
        (PREFIX_JAVASCRIPT_2, COMPLETION_JAVASCRIPT_2_1, 3, "\n\n\n"),
        (
            PREFIX_JAVASCRIPT_3,
            COMPLETION_JAVASCRIPT_3_1,
            3,
            " newFunctionForValidatingEmail, writeStringBackwards };",
        ),
        (PREFIX_JAVASCRIPT_4, COMPLETION_JAVASCRIPT_4_1, 3, COMPLETION_JAVASCRIPT_4_1),
        (PREFIX_PYTHON_1, COMPLETION_PYTHON_1_1, 3, ""),
        (PREFIX_PYTHON_2, COMPLETION_PYTHON_2_1, 3, '\n\tprint("hello world")'),
        (PREFIX_RUBY_1, COMPLETION_RUBY_1_1, 5, COMPLETION_RUBY_1_1),
        ("   ", "", 3, ""),
        ("def hello_world():", "\n\n  ", 3, "\n\n  "),
    ],
)
def test_clean_model_reflection(
    context: str, completion: str, min_block_size: int, expected: str
):
    actual = clean_model_reflection(context, completion, min_block_size=min_block_size)

    assert actual == expected
