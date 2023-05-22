#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

tempdir=$(mktemp -d)

helm template ai-assist ai-assist --debug | gawk -v "tempdir=$tempdir" '
  match($0, /# Source: ai-assist\/templates\/(.*)/, a) { 
    file=tempdir "/" a[1]
  } 
  { 
    if (file) { 
      print >file 
    } 
  }
'

for i in ${tempdir}/*.yaml; do
  base=$(basename "$i")
  target=$(find ../manifests -name "$base" -not -path "**/codegen**" | head -1)
  cp "$i" "$target"
  echo "$target"
done

yamlfmt '../manifests/**.yaml'
