#!/usr/bin/env bash

if [ -n "$WEB_CONCURRENCY" ] && [ "$WEB_CONCURRENCY" -gt 1 ]; then
  METRICS_DIR=$(mktemp -d -t ai_gateway.XXXXXX)
  echo "Storing multiprocess metrics in $METRICS_DIR..."
  export PROMETHEUS_MULTIPROC_DIR=$METRICS_DIR
fi

poetry run ai_gateway
