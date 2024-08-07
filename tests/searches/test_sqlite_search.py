import json
import os.path
from unittest.mock import patch

import pytest

from ai_gateway.searches.sqlite_search import SqliteSearch


@pytest.fixture
def mock_sqlite_search_struct_data():
    return {
        "id": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        "metadata": {
            "Header1": "Git Large File Storage (LFS)",
            "Header2": "Add a file with Git LFS",
            "Header3": "Add a file type to Git LFS",
            "filename": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        },
    }


@pytest.fixture
def mock_sqlite_search_response():
    return [
        json.dumps(
            {
                "Header1": "Tutorial: Set up issue boards for team hand-off",
                "filename": "tmp/gitlab-master-doc/doc/foo/index.md",
            }
        ),
        "GitLab's mission is to make software development easier and more efficient.",
    ]


@pytest.fixture
def sqlite_search_factory():
    def create() -> SqliteSearch:
        return SqliteSearch()

    return create


@pytest.fixture
def mock_os_path_to_db():
    current_dir = os.path.dirname(__file__)
    local_docs_example_path = current_dir.replace(
        "searches", "_assets/tpl/tools/searches/local_docs_example.db"
    )
    with patch("posixpath.join", return_value=local_docs_example_path) as mock:
        yield mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version",
    [
        "1.2.3",
        "10.11.12-pre",
    ],
)
async def test_sqlite_search(
    mock_sqlite_search_struct_data,
    mock_os_path_to_db,
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    sqlite_search = sqlite_search_factory()

    result = await sqlite_search.search(query, gl_version, page_size)
    assert result[0]["id"] == mock_sqlite_search_struct_data["id"]
    assert result[0]["metadata"] == mock_sqlite_search_struct_data["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version",
    [
        ("1.2.3"),
        ("10.11.12-pre"),
    ],
)
async def test_sqlite_search_with_no_db(
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    with patch("os.path.isfile", return_value=False):
        sqlite_search = sqlite_search_factory()

        result = await sqlite_search.search(query, gl_version, page_size)
        assert result == []
