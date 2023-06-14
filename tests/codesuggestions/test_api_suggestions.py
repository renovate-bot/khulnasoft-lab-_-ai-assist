import pytest

from codesuggestions.api.v2.endpoints.suggestions import resolve_third_party_ai_default
from codesuggestions.api.middleware import GitLabUser
from codesuggestions.config import Project


@pytest.mark.parametrize(
    "project_id,third_party_ai_default,limited_access_3rd_party_ai,exp_resolution", [
        # `project_id` exists in the `limited_access` list
        (
            123,
            True, {
                456: Project(id=456, full_name="1"),
                123: Project(id=123, full_name="1"),
                889: Project(id=889, full_name="1"),
            },
            True
        ),
        (
            123,
            False, {
                456: Project(id=456, full_name="1"),
                123: Project(id=123, full_name="1"),
                889: Project(id=889, full_name="1"),
            },
            True
        ),

        # `project_id` does not exist in the `limited_access` list
        (
            None,
            True, {
                456: Project(id=456, full_name="1"),
                123: Project(id=123, full_name="1"),
                889: Project(id=889, full_name="1"),
            },
            False
        ),
        (
            999,
            True, {
                456: Project(id=456, full_name="1"),
                123: Project(id=123, full_name="1"),
                889: Project(id=889, full_name="1"),
            },
            False
        ),
        (
            999,
            False, {
                456: Project(id=456, full_name="1"),
                123: Project(id=123, full_name="1"),
                889: Project(id=889, full_name="1"),
            },
            False
        ),
        (888, True, dict(), False),
        (888, False, dict(), False),
    ]
)
def test_resolve_third_party_ai_default(
    project_id,
    third_party_ai_default,
    limited_access_3rd_party_ai,
    exp_resolution
):
    user = GitLabUser(authenticated=True, is_debug=False)
    act = resolve_third_party_ai_default(user, project_id, third_party_ai_default, limited_access_3rd_party_ai)

    assert act == exp_resolution
