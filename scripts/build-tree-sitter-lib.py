#!/usr/bin/env python3

import contextlib
import os
import platform
import sys
from collections.abc import Generator
from pathlib import Path

from tree_sitter import Language

BASE_URL = "https://github.com/tree-sitter"

LANGS = [
    "tree-sitter-c",
    "tree-sitter-c-sharp",
    "tree-sitter-cpp",
    "tree-sitter-go",
    "tree-sitter-java",
    "tree-sitter-javascript",
    "tree-sitter-php",
    "tree-sitter-python",
    "tree-sitter-ruby",
    "tree-sitter-rust",
    "tree-sitter-scala",
    "tree-sitter-typescript",
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
        for lang in LANGS:
            if (vendor_dir / lang).exists():
                print(f"Updating {lang}")
                os.system(f"git -C {lang} pull")
            else:
                url = f"{BASE_URL}/{lang}"
                print(f"Cloning {url}")
                os.system(f"git clone {url}")

    with working_directory(vendor_dir / "tree-sitter-typescript"):
        if os.system("npm install && npm run build") != 0:
            print("error building tree-sitter-typescript")
            return 1

    language_directories = [
        str(vendor_dir / lang) for lang in LANGS if lang != "tree-sitter-typescript"
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
