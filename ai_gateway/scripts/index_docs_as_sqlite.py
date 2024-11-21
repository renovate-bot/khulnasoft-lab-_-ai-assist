# Indexes gitlab documentation to create embeddable search for self-hosted
# AIGW instances. This was initially added to GitLab-rails repository, but it
# is only used when building the docker image here.

import argparse
import json
import logging
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from shutil import rmtree
from typing import Any
from zipfile import ZipFile

import requests
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate GitLab docs index.")

    parser.add_argument("-o", "--output_path", help="Output path", default="docs.db")
    parser.add_argument(
        "-v",
        "--version_tag",
        help="GitLab version tag to include in the URL (e.g., v17.1.0-ee)",
        default="master",
    )
    return parser.parse_args()


def execution_error(error_message: str):
    logger.error(error_message)
    sys.exit(1)


# Function to fetch documents from GitLab
def fetch_documents(version_tag: str):
    docs_url = f"https://gitlab.com/gitlab-org/gitlab/-/archive/{version_tag}/gitlab-{version_tag}.zip?path=doc"
    print(docs_url)

    response = requests.get(docs_url, timeout=100)

    if response.status_code != 200:
        return execution_error(
            f"Failed to download documents. Status code: {response.status_code}"
        )

    tmpdirname = tempfile.mkdtemp()
    zip_path = Path(tmpdirname) / "docs.zip"

    with open(zip_path, "wb") as f:
        f.write(response.content)
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdirname)

    # Find the directory that was extracted
    extracted_dirs = [path for path in Path(tmpdirname).iterdir() if path.is_dir()]
    if not extracted_dirs:
        execution_error("No directory found after extraction. Exiting.")
    zip_path.unlink()
    logger.info("Documents are fetched.")
    extracted_dir = extracted_dirs[0]
    logger.info("Extracted documents to %s", extracted_dir)
    return extracted_dir


def build_row_corpus(row: dict):
    corpus = row["content"]
    # Remove the preamble
    preamble_start = corpus.find("---")
    if preamble_start != -1:
        preamble_end = corpus.find("---", preamble_start + 1)
        corpus = corpus[preamble_end + 2 : -1]
    if not corpus:
        return ""

    # Stemming could be helpful, but it is already applied by the sqlite
    # Remove punctuation and set to lowercase, this should reduce the size of the corpus and allow
    # the query to be a bit more robust
    corpus = corpus.lower()
    corpus = re.sub(r"[^\w\s]", "", corpus)
    return corpus


def process_file_to_db_entry(path: Path, file: Any) -> tuple:
    with file.open("r") as f:
        file_name = f.name.removeprefix(f"{path}/doc")
        doc = Document(page_content=f.read(), metadata={"filename": file_name})
        row = {"content": doc.page_content, "metadata": doc.metadata}

        return build_row_corpus(row), row["content"], json.dumps(row["metadata"])


def read_documents(path: Path):
    logger.info("Processing documents")
    doc_files = path.glob("doc/**/*.md")
    if not doc_files:
        execution_error("No markdown files found")

    return doc_files


def create_database(output_path: str, sql_tuples: list):
    logger.info("Creating database at %s", output_path)

    Path(output_path).unlink(True)

    # Create the database
    with closing(sqlite3.connect(output_path)) as connection:
        c = connection.cursor()
        c.execute(
            "CREATE VIRTUAL TABLE doc_index USING fts5(processed, content, metadata, tokenize='porter trigram');"
        )
        c.execute("PRAGMA user_version = 1;")
        c.executemany(
            "INSERT INTO doc_index (processed, content, metadata) VALUES (?,?,?)",
            sql_tuples,
        )
        connection.commit()


def build_indexed_docs():
    args = parse_arguments()

    docs_path = fetch_documents(version_tag=args.version_tag)

    if not docs_path:
        execution_error("Fetching documents failed")
    output_path = args.output_path
    files = read_documents(docs_path)
    sql_tuples = [process_file_to_db_entry(docs_path, file) for file in files]
    create_database(output_path, sql_tuples)
    rmtree(docs_path)
    logger.info("Database created at %s", output_path)
