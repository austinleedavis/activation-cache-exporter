#!/bin/bash

set -euo pipefail

source .env

LOCAL_BASE_PATH="data"
INCLUDE_GLOB="~/git/activation-cache-exporter/data/activations/sorted/**/*.parquet.gz"
EXCLUDE_GLOB=""  # leave empty to disable exclusion
PARALLEL_JOBS=4

export RSYNC_RSH="ssh"
mkdir -p "$LOCAL_BASE_PATH"

REMOTE_FILES=$(ssh "$REMOTE_USER@$REMOTE_HOST" bash -c "'
  shopt -s globstar nullglob
  for f in $INCLUDE_GLOB; do
    if [[ -z \"$EXCLUDE_GLOB\" || ! \$f == \$EXCLUDE_GLOB ]]; then
      echo \$f
    fi
  done
'")

echo "$REMOTE_FILES" | parallel -j "$PARALLEL_JOBS" --eta '
  rsync -ah --progress '"$REMOTE_USER"'@'"$REMOTE_HOST"':"{}" '"$LOCAL_BASE_PATH"'
'