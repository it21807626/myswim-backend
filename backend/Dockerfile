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

# Workdir
WORKDIR /app

# Install Python deps (kept separate for better cache)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the backend (app.py, models/, config/, norm/, gunicorn_start.sh, etc.)
COPY backend/ .

RUN chmod +x gunicorn_start.sh

# (Optional) container port hint
EXPOSE 5000

# Simple healthcheck hits your /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-5000}/health || exit 1

# Start with your wrapper (keeps your current worker/thread settings)
CMD ["bash", "gunicorn_start.sh"]


 
