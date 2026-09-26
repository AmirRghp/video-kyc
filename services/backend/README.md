# Backend Service

Session management, ID upload, WebSocket KYC state machine. Calls the AI service over HTTP.

## Responsibility

| Module | Function |
|--------|----------|
| `main.py` | FastAPI routes + WebSocket pipeline (unchanged logic) |
| `ai_client.py` | HTTP client mirroring original `liveness` / `challenges` / `face_match` APIs |

## Default port

`8000`

## Environment variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP listen port |
| `AI_SERVICE_URL` | `http://localhost:8001` | AI microservice base URL |
| `UPLOAD_DIR` | `uploads` | Temporary ID image storage |

## Run locally

Start the AI service first, then:

```bash
cd services/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export AI_SERVICE_URL=http://localhost:8001
uvicorn main:app --reload --port 8000 --app-dir src
```

Endpoints:

- `POST /upload_id`
- `WS /ws/kyc/{session_id}`
- `GET /health`

## Docker

```bash
docker compose up backend
```

Depends on `ai` service (see root `docker-compose.yml`).
