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

target_dir="../manifests/fauxpilot/v2"

rm -rf "${target_dir}"
mkdir "${target_dir}"

for i in ${tempdir}/*.yaml; do
  base=$(basename "$i")

  target="${target_dir}/${base}"
  echo "$i -> $target"

  tempdir2=$(mktemp -d)
  (
    cd "$tempdir2"
    yq --split-exp '.kind + "-" + $index' "$i"

    for i in *.yml; do
      echo "---"
      grep -v "^---" "$i"
    done
  ) >"$target"
  rm -rf "$tempdir2"
done

yamlfmt "${target_dir}/**.yaml"
