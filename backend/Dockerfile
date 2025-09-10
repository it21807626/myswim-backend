# ---- Base: pick a TF/Keras sweet spot for legacy .h5 ----
FROM python:3.10-slim

# System deps for OpenCV / MediaPipe / video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create workspace
WORKDIR /app

# Copy requirement pins first (better Docker layer cache)
COPY requirements.txt /app/

# Install Python deps
RUN pip install --upgrade pip==23.2.1 && \
    pip install -r requirements.txt

# Copy the rest of your backend (app.py, models/, config/, norm/, etc.)
COPY . /app/

# Render sets $PORT for us; default to 5000 locally
ENV PORT=5000

# Gunicorn entry (use threads for I/O bound)
CMD exec gunicorn -w 2 -k gthread -t 180 -b 0.0.0.0:${PORT} app:app

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Gunicorn serving Flask app: app.py exposes 'app'
CMD gunicorn -b 0.0.0.0:$PORT -w 2 -k gthread app:app
 
