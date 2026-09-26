# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — download MediaPipe model
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS model-fetch

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /models
RUN wget -qO face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# MediaPipe needs GLES/EGL shared libs (libGLESv2.so.2) at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download VGG-Face weights so first user request does not hang on ~145MB download
RUN mkdir -p /root/.deepface/weights \
    && wget -qO /root/.deepface/weights/vgg_face_weights.h5 \
        https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

COPY src/ ./src/
COPY --from=model-fetch /models/face_landmarker.task ./src/face_landmarker.task

ENV PYTHONPATH=/app/src
ENV PORT=8001

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "/app/src"]
