# AI Service

MediaPipe liveness (EAR blink, landmarks), active challenges (smile / head yaw), and DeepFace face matching.

## Responsibility

| Module | Function |
|--------|----------|
| `liveness.py` | Face landmarks + EAR blink detection |
| `challenges.py` | Smile and head-turn challenge detectors |
| `face_match.py` | VGG-Face comparison via DeepFace |
| `app.py` | HTTP API wrapping the modules above |

## Default port

`8001`

## Environment variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8001` | HTTP listen port |
| `DEEPFACE_HOME` | `~/.deepface` | DeepFace model cache directory |

## Run locally

```bash
cd services/ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download MediaPipe model (once)
wget -qO src/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

uvicorn app:app --reload --port 8001 --app-dir src
```

Health: `http://localhost:8001/health`

## Docker

```bash
docker compose build ai
docker compose up ai
```

From project root with full stack:

```bash
docker compose up --build
```
