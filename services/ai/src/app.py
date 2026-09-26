"""
app.py — AI Service HTTP API

Exposes the existing liveness, challenge, and face-match modules over REST.
No business-logic changes — thin wrappers only.
"""

import base64
import os
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from challenges import detect_head_yaw, detect_smile
from liveness import detect_ear, detect_landmarks

app = FastAPI(title="Video KYC AI Service")


class FramePayload(BaseModel):
    frame_b64: str


class LandmarksPayload(BaseModel):
    landmarks: list[dict]
    w: int
    h: int


class ChallengePayload(BaseModel):
    frame_b64: str
    challenge: str


def _decode_frame(frame_b64: str) -> np.ndarray | None:
    raw = base64.b64decode(frame_b64)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return frame


def _serialize_landmarks(landmarks) -> list[dict]:
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]


def _deserialize_landmarks(data: list[dict]) -> list:
    return [SimpleNamespace(x=lm["x"], y=lm["y"], z=lm.get("z", 0)) for lm in data]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect-ear")
async def api_detect_ear(payload: FramePayload):
    frame = _decode_frame(payload.frame_b64)
    if frame is None:
        return {"ear": None}
    ear = detect_ear(frame)
    return {"ear": ear}


@app.post("/api/detect-landmarks")
async def api_detect_landmarks(payload: FramePayload):
    frame = _decode_frame(payload.frame_b64)
    if frame is None:
        return {"landmarks": None}
    landmarks = detect_landmarks(frame)
    if landmarks is None:
        return {"landmarks": None}
    return {"landmarks": _serialize_landmarks(landmarks)}


@app.post("/api/detect-smile")
async def api_detect_smile(payload: LandmarksPayload):
    landmarks = _deserialize_landmarks(payload.landmarks)
    detected = detect_smile(landmarks, payload.w, payload.h)
    return {"detected": detected}


@app.post("/api/detect-head-yaw")
async def api_detect_head_yaw(payload: LandmarksPayload):
    landmarks = _deserialize_landmarks(payload.landmarks)
    yaw = detect_head_yaw(landmarks, payload.w, payload.h)
    return {"yaw": yaw}


@app.post("/api/evaluate-challenge")
async def api_evaluate_challenge(payload: ChallengePayload):
    """Single inference pass: landmarks + challenge action (avoids duplicate HTTP round-trips)."""
    frame = _decode_frame(payload.frame_b64)
    if frame is None:
        return {"face_visible": False, "action_detected": False}

    landmarks = detect_landmarks(frame)
    if landmarks is None:
        return {"face_visible": False, "action_detected": False}

    h, w, _ = frame.shape
    challenge = payload.challenge

    if challenge == "SMILE":
        action_detected = detect_smile(landmarks, w, h)
    elif challenge == "TURN_LEFT":
        action_detected = detect_head_yaw(landmarks, w, h) == "LEFT"
    elif challenge == "TURN_RIGHT":
        action_detected = detect_head_yaw(landmarks, w, h) == "RIGHT"
    else:
        action_detected = False

    return {"face_visible": True, "action_detected": action_detected}


@app.post("/api/compare-faces")
async def api_compare_faces(
    id_image: UploadFile = File(...),
    live_frame: UploadFile = File(...),
):
    id_bytes = await id_image.read()
    live_bytes = await live_frame.read()

    live_arr = cv2.imdecode(np.frombuffer(live_bytes, np.uint8), cv2.IMREAD_COLOR)
    if live_arr is None:
        return {
            "success": False,
            "error": "Could not decode live frame.",
        }

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(id_bytes)
        id_path = tmp.name

    try:
        from face_match import compare_faces

        result = compare_faces(id_path, live_arr)
    finally:
        if os.path.exists(id_path):
            os.remove(id_path)

    return result
