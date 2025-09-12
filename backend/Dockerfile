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

# Copy everything and run from repo root (where app.py is)
WORKDIR /app
COPY . .

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-5000}/health || exit 1

# Run Gunicorn via python -m (avoids PATH issues)
CMD ["bash","-lc","exec python -m gunicorn app:app -w 1 -k gthread --threads 8 -b 0.0.0.0:${PORT:-5000} --timeout 180"]



 
