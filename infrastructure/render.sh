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

function find_target_for() {
  if [[ -f ../manifests/fauxpilot/v2/$1 ]]; then
    echo ../manifests/fauxpilot/v2/$1
    return
  fi

  if [[ -f ../manifests/fauxpilot/$1 ]]; then
    echo ../manifests/fauxpilot/$1
    return
  fi

  find ../manifests -name "$base" -not -path "**/codegen**" | head -1
}

git restore --source=origin/main --staged --worktree -- ../manifests

for i in ${tempdir}/*.yaml; do
  base=$(basename "$i")

  target=$(find_target_for "${base}")
  echo "$i -> $target"

  tempdir2=$(mktemp -d)
  (
    cd "$tempdir2"
    yq --split-exp '.kind + "-" + $index' "$i"
    cat *.yml
  ) >"$target"
  rm -rf "$tempdir2"

  # cp "$i" "$target"
done

yamlfmt '../manifests/**.yaml'
