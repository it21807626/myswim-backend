#!/usr/bin/env bash
set -euo pipefail

# sensible defaults, can be overridden by Render env vars
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

HOST="0.0.0.0"
PORT="${PORT:-5000}"

# Run a single worker + single thread
exec gunicorn app:app \
  --bind "${HOST}:${PORT}" \
  --workers 1 \
  --threads 1 \
  --worker-class sync \
  --timeout 120


