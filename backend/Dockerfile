FROM python:3.10-slim

# OS deps for OpenCV/MediaPipe/video I/O + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

# ---- Install Python deps for BACKEND ONLY (cache-friendly) ----
WORKDIR /app/backend
COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Copy the backend code/models/config into the image ----
COPY backend/ ./

# (optional) expose & healthcheck
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-5000}/health || exit 1

# ---- Start the app: run Gunicorn via python -m ----
CMD ["bash","-lc","exec python -m gunicorn app:app -w 1 -k gthread --threads 8 -b 0.0.0.0:${PORT:-5000} --timeout 180"]



 
