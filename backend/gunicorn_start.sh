#!/usr/bin/env bash
set -euo pipefail

# Keep TensorFlow single-threaded & quiet
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# Absolutely force single worker + single thread
exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers 1 \
  --worker-class sync \
  --threads 1 \
  --timeout 120 \
  --preload=false

