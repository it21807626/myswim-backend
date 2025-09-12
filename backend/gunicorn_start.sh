#!/usr/bin/env bash
set -euo pipefail

# Keep CPU usage and memory stable
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

# Gunicorn runtime knobs (overridable via env)
WORKERS="${WORKERS:-1}"
THREADS="${THREADS:-1}"
TIMEOUT="${TIMEOUT:-180}"
MAX_REQUESTS="${MAX_REQUESTS:-5}"
MAX_REQUESTS_JITTER="${MAX_REQUESTS_JITTER:-3}"

# Single worker/thread; recycle to avoid leaks; longer timeout for video analysis
exec python -m gunicorn app:app \
  --bind "${HOST}:${PORT}" \
  --workers "${WORKERS}" \
  --threads "${THREADS}" \
  --worker-class sync \
  --timeout "${TIMEOUT}" \
  --max-requests "${MAX_REQUESTS}" \
  --max-requests-jitter "${MAX_REQUESTS_JITTER}"


