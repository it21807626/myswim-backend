#!/usr/bin/env bash
set -e
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-120}"

exec gunicorn app:app \
  --bind 0.0.0.0:"$PORT" \
  --workers "$WORKERS" \
  --threads "$THREADS" \
  --timeout "$TIMEOUT"

