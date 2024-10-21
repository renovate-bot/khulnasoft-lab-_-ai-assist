import pytest

from ai_gateway.code_suggestions.processing.post.ops import strip_asterisks

# a completion has asterisks at the start, with legitimate suggestions after it
COMPLETION_RUBY_1_1 = """
****************************
    expect(person.fullname).to eq('First')
  end
""".strip(
    "\n"
)

# a completion only has asterisks (ignoring leading spaces)
COMPLETION_RUBY_1_2 = "\n\t\t    ****************************\nend"

# a completion has asterisks after some legitimate suggestions
# this does not happen in practice, so we are not addressing it in the
# current `strip_asterisks` logic
COMPLETION_RUBY_1_3 = """
    expect(person.fullname).to eq('First')
    ****************************
""".strip(
    "\n"
)

# a completion has no asterisks
COMPLETION_RUBY_1_4 = "expect(person.fullname).to eq('First')"

COMPLETION_VUE_1 = """
      ***********="true"
    >
""".strip(
    "\n"
)

COMPLETION_VUE_2 = "\n\t\t    *****************>"

COMPLETION_VUE_3 = """
      modal="true"
      *****************
    >
"""

COMPLETION_VUE_4 = """
      modal="true"
      title="Hello World"
    >
"""


@pytest.mark.parametrize(
    ("completion", "expected_result"),
    [
        (COMPLETION_RUBY_1_1, ""),
        (COMPLETION_RUBY_1_2, ""),
        (COMPLETION_RUBY_1_3, COMPLETION_RUBY_1_3),
        (COMPLETION_RUBY_1_4, COMPLETION_RUBY_1_4),
        (COMPLETION_VUE_1, ""),
        (COMPLETION_VUE_2, ""),
        (COMPLETION_VUE_3, COMPLETION_VUE_3),
        (COMPLETION_VUE_4, COMPLETION_VUE_4),
    ],
)
def test_strip_asterisks(completion: str, expected_result: str):
    actual_result = strip_asterisks(completion)

    assert actual_result == expected_result
