# ---- Base: Python 3.10 works well with TF/Keras + your .h5 files
FROM python:3.10-slim

# OS deps for OpenCV/MediaPipe/video I/O
# + libgomp1 (OpenMP) for some TF/Mediapipe ops
# + curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

# Copy everything, then run from repo root
WORKDIR /app
COPY . .

# If your script is at backend/gunicorn_start.sh (most likely):
RUN chmod +x backend/gunicorn_start.sh

# Healthcheck
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-5000}/health || exit 1

# Start: cd into backend and run the script
CMD ["bash", "-lc", "cd backend && ./gunicorn_start.sh"]


 
