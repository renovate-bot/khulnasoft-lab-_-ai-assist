#!/usr/bin/env python3

from tree_sitter import Language
import os

BASE_URL = 'https://github.com/tree-sitter/'

LANGS = [
    'tree-sitter-c',
    'tree-sitter-c-sharp',
    'tree-sitter-cpp',
    'tree-sitter-go',
    'tree-sitter-java',
    'tree-sitter-javascript',
    'tree-sitter-php',
    'tree-sitter-python',
    'tree-sitter-rust',
    'tree-sitter-scala',
    'tree-sitter-typescript'
]

scripts_dir = os.path.dirname(__file__)
vendor_dir = os.path.join(scripts_dir, 'vendor')
lib_dir = os.path.realpath(os.path.join(scripts_dir, '..', 'lib'))
print("Checking out grammars in %s" % vendor_dir)

if not os.path.exists(vendor_dir):
    os.makedirs(vendor_dir)

os.chdir(vendor_dir)

for lang in LANGS:
    if os.path.exists(lang):
        print("Updating %s" % lang)
        os.system("git -C %s pull" % lang)
    else:
        url = "%s/%s" % (BASE_URL, lang)
        print("Cloning %s" % url)
        os.system("git clone %s" % url)

os.chdir(os.path.join(vendor_dir, "tree-sitter-typescript"))
if os.system("npm install && npm run build") != 0:
    print('error building tree-sitter-typescript')
    exit(1)

language_directories = ["%s" % (os.path.join(vendor_dir, lang)) for lang in LANGS if lang != 'tree-sitter-typescript']
language_directories += [os.path.join(vendor_dir, 'tree-sitter-typescript/typescript'),
                         os.path.join(vendor_dir, 'tree-sitter-typescript/tsx')]

lib = os.path.join(lib_dir, 'tree-sitter-languages.so')
print("Building %s" % lib)
Language.build_library(lib, language_directories)
