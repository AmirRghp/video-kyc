# Video KYC — Microservices

Video Identity Verification (KYC) MVP split into three Dockerized services.

---

## مرحله ۱ — گزارش کشف پروژه

### فایل‌های اصلی (قبل از refactor)

| مسیر قبلی | نقش |
|-----------|-----|
| `main.py` | FastAPI: آپلود مدارک، WebSocket، state machine |
| `liveness.py` | MediaPipe — EAR و landmarks |
| `challenges.py` | لبخند و چرخش سر |
| `face_match.py` | DeepFace VGG-Face |
| `templates/index.html` | UI تک‌صفحه‌ای |
| `requirements.txt` | وابستگی‌های Python |
| `face_landmarker.task` | وزن MediaPipe (دانلود جداگانه) |

### مرزهای منطقی

| لایه | محتوا |
|------|--------|
| **Frontend** | HTML/CSS/JS — دوربین، آپلود، WebSocket |
| **Backend** | نشست، آپلود فایل، orchestration |
| **AI** | MediaPipe + DeepFace (پردازش تصویر) |

### وابستگی‌های مشترک (قبلاً in-process)

- `main.py` → `liveness`, `challenges`, `face_match`
- `challenges` → landmarks از `liveness` (همان inference)

پس از refactor: **backend → HTTP → ai**

---

## مرحله ۲ — تعریف سرویس‌ها

| سرویس | مسئولیت | پورت | فناوری | وابستگی |
|--------|---------|------|---------|----------|
| **frontend** | UI استاتیک + reverse proxy | `3000` | nginx + HTML | backend |
| **backend** | API + WebSocket + sessions | `8000` | FastAPI | ai |
| **ai** | liveness, challenges, face match | `8001` | FastAPI + MediaPipe + DeepFace | — |

---

## ساختار پروژه

```
video_kyc/
├── services/
│   ├── ai/
│   │   ├── src/
│   │   │   ├── app.py
│   │   │   ├── liveness.py
│   │   │   ├── challenges.py
│   │   │   └── face_match.py
│   │   ├── Dockerfile
│   │   ├── .env.example
│   │   └── README.md
│   ├── backend/
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   └── ai_client.py
│   │   ├── Dockerfile
│   │   ├── .env.example
│   │   └── README.md
│   └── frontend/
│       ├── src/
│       │   └── index.html
│       ├── nginx.conf
│       ├── Dockerfile
│       ├── .env.example
│       └── README.md
├── docker-compose.yml
├── docker-compose.dev.yml
└── README.md
```

### نقشه مهاجرت فایل‌ها

| قبلی | جدید |
|------|------|
| `main.py` | `services/backend/src/main.py` |
| `liveness.py` | `services/ai/src/liveness.py` |
| `challenges.py` | `services/ai/src/challenges.py` |
| `face_match.py` | `services/ai/src/face_match.py` |
| `templates/index.html` | `services/frontend/src/index.html` |
| — (جدید) | `services/ai/src/app.py` |
| — (جدید) | `services/backend/src/ai_client.py` |

---

## Quick start (Docker)

```bash
# Production-like stack
docker compose up --build

# Development (hot reload + exposed ports)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Open: **http://localhost:3000**

> First AI container start may take several minutes (DeepFace downloads VGG-Face ~145 MB into the `deepface-cache` volume).

---

## Quick start (local, without Docker)

```bash
# Terminal 1 — AI
cd services/ai && source .venv/bin/activate  # create venv + pip install first
wget -qO src/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
uvicorn app:app --port 8001 --app-dir src

# Terminal 2 — Backend
cd services/backend && source .venv/bin/activate
export AI_SERVICE_URL=http://localhost:8001
uvicorn main:app --port 8000 --app-dir src

# Terminal 3 — Full stack via Docker frontend only, or use dev compose
docker compose -f docker-compose.yml -f docker-compose.dev.yml up frontend
```

---

## Networks & volumes (Docker)

| Network | Members | Purpose |
|---------|---------|---------|
| `edge` | frontend, backend | Public-facing proxy path |
| `ai-internal` | backend, ai (internal) | AI isolated from host |

| Volume | Purpose |
|--------|---------|
| `uploads` | Temporary ID images |
| `deepface-cache` | Persist DeepFace model weights |

---

## KYC pipeline (unchanged logic)

1. **Blink** — 3× EAR blink (MediaPipe)
2. **Challenges** — 2 random actions, 3s hold each, 10s timeout
3. **Face match** — VGG-Face vs uploaded ID

---

## Per-service docs

- [AI Service](services/ai/README.md)
- [Backend](services/backend/README.md)
- [Frontend](services/frontend/README.md)
