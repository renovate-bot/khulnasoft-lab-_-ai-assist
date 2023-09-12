import pytest

from ai_gateway.suggestions.processing.post.ops import clean_model_reflection

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


@pytest.mark.parametrize(
    ("context", "completion", "expected"),
    [
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_1, "only\n"),
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_2, "only"),
        (PREFIX_JAVASCRIPT_1, COMPLETION_JAVASCRIPT_1_3, COMPLETION_JAVASCRIPT_1_3),
        (PREFIX_JAVASCRIPT_2, COMPLETION_JAVASCRIPT_2_1, "\n\n\n"),
        (
            PREFIX_JAVASCRIPT_3,
            COMPLETION_JAVASCRIPT_3_1,
            " newFunctionForValidatingEmail, writeStringBackwards };",
        ),
        (PREFIX_PYTHON_1, COMPLETION_PYTHON_1_1, ""),
        (PREFIX_PYTHON_2, COMPLETION_PYTHON_2_1, '\n\tprint("hello world")'),
        ("   ", "", ""),
        ("def hello_world():", "\n\n  ", "\n\n  "),
    ],
)
def test_clean_model_reflection(context: str, completion: str, expected: str):
    actual = clean_model_reflection(context, completion)

    assert actual == expected
