# ---- Base: Python 3.10 works well with TF/Keras + your .h5 files
FROM python:3.10-slim

# OS deps for OpenCV/MediaPipe/video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

# Workdir
WORKDIR /app

# Install Python deps (keep this separate for better cache)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the backend (app.py, models/, config/, norm/, gunicorn_start.sh, etc.)
COPY . .

RUN chmod +x gunicorn_start.sh


# Start with our wrapper (forces workers=1/threads=1)
CMD ["bash", "gunicorn_start.sh"]

 
