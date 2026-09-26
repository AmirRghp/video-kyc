# Frontend Service

Single-page Video KYC UI served by nginx. Proxies API and WebSocket traffic to the backend.

## Responsibility

| File | Function |
|------|----------|
| `src/index.html` | Camera selection, ID upload, WebSocket frame streaming |
| `nginx.conf` | Static files + reverse proxy to `backend:8000` |

## Default port

`3000` (mapped to nginx port 80 in Docker)

## Environment variables

Copy `.env.example` to `.env` (used by docker-compose for port mapping):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_PORT` | `3000` | Host port exposed by Docker |

## Run locally (without Docker)

Serve `src/index.html` via any static server **and** ensure `/upload_id` and `/ws/` reach the backend, **or** open the app through the full stack:

```bash
# From project root
docker compose -f docker-compose.yml -f docker-compose.dev.yml up frontend backend ai
```

Then open: `http://localhost:3000`

## Docker

```bash
docker compose up frontend
```

Requires `backend` (and indirectly `ai`) to be healthy.
