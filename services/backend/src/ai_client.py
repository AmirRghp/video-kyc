"""
ai_client.py — HTTP client for the AI service.

Provides the same function signatures as the original in-process modules
so main.py state-machine logic stays unchanged.
"""

import base64
import os
from types import SimpleNamespace

import cv2
import httpx
import numpy as np

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")

_client = httpx.Client(base_url=AI_SERVICE_URL, timeout=120.0)


class LandmarkProxy:
    """Minimal stand-in for MediaPipe NormalizedLandmark (.x / .y / .z)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def _serialize_landmarks(landmarks) -> list[dict]:
    return [{"x": lm.x, "y": lm.y, "z": getattr(lm, "z", 0)} for lm in landmarks]


def detect_ear(frame) -> float | None:
    resp = _client.post(
        "/api/detect-ear",
        json={"frame_b64": _encode_frame(frame)},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("ear")


def detect_landmarks(frame) -> list | None:
    resp = _client.post(
        "/api/detect-landmarks",
        json={"frame_b64": _encode_frame(frame)},
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("landmarks")
    if raw is None:
        return None
    return [LandmarkProxy(lm["x"], lm["y"], lm.get("z", 0)) for lm in raw]


def detect_smile(landmarks, w: int, h: int) -> bool:
    resp = _client.post(
        "/api/detect-smile",
        json={"landmarks": _serialize_landmarks(landmarks), "w": w, "h": h},
    )
    resp.raise_for_status()
    return resp.json()["detected"]


def detect_head_yaw(landmarks, w: int, h: int) -> str | None:
    resp = _client.post(
        "/api/detect-head-yaw",
        json={"landmarks": _serialize_landmarks(landmarks), "w": w, "h": h},
    )
    resp.raise_for_status()
    return resp.json().get("yaw")


def evaluate_challenge(frame, challenge: str) -> tuple[bool, bool]:
    """Returns (face_visible, action_detected) in a single AI round-trip."""
    resp = _client.post(
        "/api/evaluate-challenge",
        json={"frame_b64": _encode_frame(frame), "challenge": challenge},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("face_visible", False), data.get("action_detected", False)


def compare_faces(id_img_path: str, live_frame) -> dict:
    with open(id_img_path, "rb") as f:
        id_bytes = f.read()
    _, buf = cv2.imencode(".jpg", live_frame)
    live_bytes = buf.tobytes()

    resp = _client.post(
        "/api/compare-faces",
        files={
            "id_image": ("id.jpg", id_bytes, "image/jpeg"),
            "live_frame": ("live.jpg", live_bytes, "image/jpeg"),
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()
