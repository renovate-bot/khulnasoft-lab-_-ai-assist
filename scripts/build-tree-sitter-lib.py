#!/usr/bin/env python3

import contextlib
import os
import platform
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

from tree_sitter import Language

LANGS = [
    (
        "tree-sitter-c",
        "https://github.com/tree-sitter/tree-sitter-c",
        "25371f9448b97c55b853a6ee8bb0bfb1bca6da9f",
    ),
    (
        "tree-sitter-c-sharp",
        "https://github.com/tree-sitter/tree-sitter-c-sharp",
        "9c494a503c8e2044bfffce57f70b480c01a82f03",
    ),
    (
        "tree-sitter-cpp",
        "https://github.com/tree-sitter/tree-sitter-cpp",
        "a90f170f92d5d70e7c2d4183c146e61ba5f3a457",
    ),
    (
        "tree-sitter-go",
        "https://github.com/tree-sitter/tree-sitter-go",
        "bbaa67a180cfe0c943e50c55130918be8efb20bd",
    ),
    (
        "tree-sitter-java",
        "https://github.com/tree-sitter/tree-sitter-java",
        "2b57cd9541f9fd3a89207d054ce8fbe72657c444",
    ),
    (
        "tree-sitter-javascript",
        "https://github.com/tree-sitter/tree-sitter-javascript",
        "f1e5a09b8d02f8209a68249c93f0ad647b228e6e",
    ),
    (
        "tree-sitter-kotlin",
        "https://github.com/fwcd/tree-sitter-kotlin",
        "16de60e6588ad39afe274b13cd494f97e0f953c7",
    ),
    (
        "tree-sitter-php",
        "https://github.com/tree-sitter/tree-sitter-php",
        "33e30169e6f9bb29845c80afaa62a4a87f23f6d6",
    ),
    (
        "tree-sitter-python",
        "https://github.com/tree-sitter/tree-sitter-python",
        "82f5c9937fe4300b4bec3ee0e788d642c77aab2c",
    ),
    (
        "tree-sitter-ruby",
        "https://github.com/tree-sitter/tree-sitter-ruby",
        "f257f3f57833d584050336921773738a3fd8ca22",
    ),
    (
        "tree-sitter-rust",
        "https://github.com/tree-sitter/tree-sitter-rust",
        "48e053397b587de97790b055a1097b7c8a4ef846",
    ),
    (
        "tree-sitter-scala",
        "https://github.com/tree-sitter/tree-sitter-scala",
        "1b4c2fa5c55c5fd83cbb0d2f818f916aba221a42",
    ),
    (
        "tree-sitter-typescript",
        "https://github.com/tree-sitter/tree-sitter-typescript",
        "d847898fec3fe596798c9fda55cb8c05a799001a",
    ),
]


@contextlib.contextmanager
def working_directory(path: Path) -> Generator[None, None, None]:
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def main() -> int:
    """Clone and build treesitter language libraries."""
    scripts_dir = Path(__file__).resolve().parent
    vendor_dir = scripts_dir / "vendor" / platform.platform()
    lib_dir = scripts_dir / "lib"
    print(f"Checking out grammars in {vendor_dir}")

    if not vendor_dir.exists():
        vendor_dir.mkdir(parents=True)

    with working_directory(vendor_dir):
        for name, url, commit in LANGS:
            if (vendor_dir / name).exists():
                print(f"Updating {name}")

                subprocess.check_call(["git", "checkout", commit], cwd=name)
            else:
                print(f"Initializing {name}")

                os.mkdir(name)
                subprocess.check_call(["git", "init"], cwd=name)
                subprocess.check_call(["git", "remote", "add", "origin", url], cwd=name)
                subprocess.check_call(
                    ["git", "fetch", "--depth=1", "origin", commit], cwd=name
                )
                subprocess.check_call(["git", "checkout", commit], cwd=name)

    language_directories = [
        str(vendor_dir / name)
        for name, _, _ in LANGS
        if name != "tree-sitter-typescript"
    ]
    language_directories += [
        str(vendor_dir / "tree-sitter-typescript/typescript"),
        str(vendor_dir / "tree-sitter-typescript/tsx"),
    ]

    lib = str(lib_dir / "tree-sitter-languages.so")
    print(f"Building {lib}")
    Language.build_library(lib, language_directories)

    return 0


if __name__ == "__main__":
    sys.exit(main())
