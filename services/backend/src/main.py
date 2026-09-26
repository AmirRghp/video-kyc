"""
main.py — Video KYC Backend (orchestration)

Pipeline (4 steps):
    1. Blink Detection    — 3 blinks via EAR
    2. Active Challenges  — 2 random actions (smile / turn left / turn right), 5s each
    3. Face Matching       — DeepFace VGG-Face comparison against uploaded ID

Endpoints:
    POST /upload_id     → Accept an ID photo + create a session
    WS   /ws/kyc/{id}   → Real-time liveness + challenge + face-match pipeline
"""

import asyncio
import base64
import logging
import os
import random
import shutil
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect

from ai_client import compare_faces, detect_ear, evaluate_challenge

# ---------------------------------------------------------------------------
# App & Config
# ---------------------------------------------------------------------------
app = FastAPI(title="Video KYC Backend")
logger = logging.getLogger("uvicorn.error")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store
sessions: dict[str, dict] = {}

# -- Blink detection config --
EAR_THRESH = 0.25
CONSEC_FRAMES = 1
REQUIRED_BLINKS = 3

# -- Challenge config --
CHALLENGE_POOL = ["SMILE", "TURN_LEFT", "TURN_RIGHT"]
NUM_CHALLENGES = 2
CHALLENGE_TIMEOUT = 10.0   # seconds to complete each challenge (including hold time)
CHALLENGE_HOLD = 3.0       # seconds the user must HOLD the action continuously

# Human-readable labels sent to the frontend
CHALLENGE_LABELS = {
    "SMILE": "😊 Smile clearly!",
    "TURN_LEFT": "👈 Turn your head LEFT",
    "TURN_RIGHT": "👉 Turn your head RIGHT",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_id")
async def upload_id(file: UploadFile = File(...)):
    """Accept an ID photo and return a unique session token."""
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate a random challenge sequence (2 out of 3, no duplicates)
    challenge_seq = random.sample(CHALLENGE_POOL, NUM_CHALLENGES)

    sessions[session_id] = {
        "id_path": file_path,
        # Step 1: Blink detection
        "blink_frames": 0,
        "blink_count": 0,
        "blinked": False,
        # Step 2: Active challenges
        "challenges": challenge_seq,
        "challenge_idx": 0,
        "challenge_start": None,
        "challenge_hold_start": None,  # when user started holding the action
        "challenges_done": False,
        # Step 3: Face matching
        "verified": False,
        "matching_started": False,
    }
    return {"session_id": session_id}


# ---------------------------------------------------------------------------
# WebSocket KYC Pipeline
# ---------------------------------------------------------------------------

@app.websocket("/ws/kyc/{session_id}")
async def kyc_websocket(websocket: WebSocket, session_id: str):
    """
    Real-time KYC state-machine:
        Step 1 → Blink-based liveness (3 blinks)
        Step 2 → Active challenge-response (smile / head turns)
        Step 3 → Face comparison against uploaded ID
    """
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({
            "status": "error",
            "message": "Session expired or invalid. Please re-upload your ID.",
        })
        await websocket.close()
        return

    session = sessions[session_id]
    frame_busy = False

    try:
        while True:
            data = await websocket.receive_text()

            if frame_busy:
                continue

            # --- Decode base64 JPEG frame ---
            try:
                _header, encoded = data.split(",", 1)
                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(encoded), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    continue
            except (ValueError, Exception):
                await websocket.send_json({
                    "status": "error",
                    "message": "Corrupted frame received.",
                })
                continue

            frame_busy = True
            try:
                await _process_frame(websocket, session, frame)
            except Exception as e:
                logger.exception("Frame processing failed for session %s", session_id)
                await websocket.send_json({
                    "status": "error",
                    "message": f"Processing error: {e}",
                })
            finally:
                frame_busy = False

    except WebSocketDisconnect:
        logger.info("Client disconnected → %s", session_id)

    finally:
        _cleanup_session(session_id)


async def _process_frame(websocket: WebSocket, session: dict, frame) -> None:
    """Run one frame through the KYC state machine (AI calls off the event loop)."""

    # ==============================================================
    # STEP 1 — Blink Detection (3× blink)
    # ==============================================================
    if not session["blinked"]:
        ear = await asyncio.to_thread(detect_ear, frame)

        if ear is None:
            await websocket.send_json({
                "status": "processing",
                "message": "Align your face inside the camera frame.",
            })
            return

        if ear < EAR_THRESH:
            session["blink_frames"] += 1
        else:
            if session["blink_frames"] >= CONSEC_FRAMES:
                session["blink_count"] += 1
                if session["blink_count"] >= REQUIRED_BLINKS:
                    session["blinked"] = True
                    session["challenge_start"] = time.time()
                    first_challenge = session["challenges"][0]
                    await websocket.send_json({
                        "status": "liveness_passed",
                        "message": f"All {REQUIRED_BLINKS} blinks confirmed!",
                    })
                    await websocket.send_json({
                        "status": "challenge",
                        "message": CHALLENGE_LABELS[first_challenge],
                        "challenge": first_challenge,
                        "challenge_num": 1,
                        "total_challenges": NUM_CHALLENGES,
                        "timeout": CHALLENGE_TIMEOUT,
                    })
                else:
                    await websocket.send_json({
                        "status": "processing",
                        "message": f"Blink {session['blink_count']}/{REQUIRED_BLINKS} detected. Keep blinking…",
                    })
            session["blink_frames"] = 0

        if not session["blinked"]:
            await websocket.send_json({
                "status": "processing",
                "message": f"Blink naturally ({session['blink_count']}/{REQUIRED_BLINKS}). EAR: {ear:.2f}",
            })

    # ==============================================================
    # STEP 2 — Active Challenge-Response (with 3s hold)
    # ==============================================================
    elif not session["challenges_done"]:
        idx = session["challenge_idx"]
        current_challenge = session["challenges"][idx]
        elapsed = time.time() - session["challenge_start"]

        # --- Timeout check ---
        if elapsed > CHALLENGE_TIMEOUT:
            await websocket.send_json({
                "status": "challenge_timeout",
                "message": f"⏰ Time's up! Failed: {CHALLENGE_LABELS[current_challenge]}",
                "challenge": current_challenge,
            })
            await websocket.close()
            return

        face_visible, action_detected = await asyncio.to_thread(
            evaluate_challenge, frame, current_challenge
        )

        if not face_visible:
            session["challenge_hold_start"] = None  # reset hold
            remaining = max(0, CHALLENGE_TIMEOUT - elapsed)
            await websocket.send_json({
                "status": "challenge",
                "message": f"{CHALLENGE_LABELS[current_challenge]} (face not visible)",
                "challenge": current_challenge,
                "challenge_num": idx + 1,
                "total_challenges": NUM_CHALLENGES,
                "remaining": round(remaining, 1),
                "hold_progress": 0.0,
            })
            return

        now = time.time()
        remaining = max(0, CHALLENGE_TIMEOUT - elapsed)

        if action_detected:
            if session["challenge_hold_start"] is None:
                session["challenge_hold_start"] = now

            held_for = now - session["challenge_hold_start"]
            hold_remaining = max(0, CHALLENGE_HOLD - held_for)

            if held_for >= CHALLENGE_HOLD:
                # ✅ Challenge passed!
                session["challenge_hold_start"] = None
                session["challenge_idx"] += 1

                if session["challenge_idx"] >= NUM_CHALLENGES:
                    session["challenges_done"] = True
                    await websocket.send_json({
                        "status": "challenge_passed",
                        "message": f"✅ Challenge {idx + 1}/{NUM_CHALLENGES} passed! All done.",
                        "challenge": current_challenge,
                    })
                else:
                    next_ch = session["challenges"][session["challenge_idx"]]
                    session["challenge_start"] = time.time()
                    await websocket.send_json({
                        "status": "challenge_passed",
                        "message": f"✅ Challenge {idx + 1}/{NUM_CHALLENGES} passed!",
                        "challenge": current_challenge,
                    })
                    await websocket.send_json({
                        "status": "challenge",
                        "message": CHALLENGE_LABELS[next_ch],
                        "challenge": next_ch,
                        "challenge_num": session["challenge_idx"] + 1,
                        "total_challenges": NUM_CHALLENGES,
                        "timeout": CHALLENGE_TIMEOUT,
                        "hold_progress": 0.0,
                    })
            else:
                # Holding — send progress
                await websocket.send_json({
                    "status": "challenge",
                    "message": f"{CHALLENGE_LABELS[current_challenge]} — hold {hold_remaining:.1f}s",
                    "challenge": current_challenge,
                    "challenge_num": idx + 1,
                    "total_challenges": NUM_CHALLENGES,
                    "remaining": round(remaining, 1),
                    "hold_progress": round(held_for / CHALLENGE_HOLD, 2),
                })
        else:
            # Action lost — reset hold timer
            session["challenge_hold_start"] = None
            await websocket.send_json({
                "status": "challenge",
                "message": CHALLENGE_LABELS[current_challenge],
                "challenge": current_challenge,
                "challenge_num": idx + 1,
                "total_challenges": NUM_CHALLENGES,
                "remaining": round(remaining, 1),
                "hold_progress": 0.0,
            })

    # ==============================================================
    # STEP 3 — Face Matching (once per session — DeepFace is slow on CPU)
    # ==============================================================
    elif not session["verified"]:
        if session.get("matching_started"):
            return

        session["matching_started"] = True
        await websocket.send_json({
            "status": "matching",
            "message": "Comparing face with ID… this may take up to a minute on first run.",
        })

        result = await asyncio.to_thread(compare_faces, session["id_path"], frame)

        if not result["success"]:
            await websocket.send_json({
                "status": "error",
                "message": f"Analyzer Error: {result.get('error')}",
            })
            return

        if result["verified"]:
            await websocket.send_json({
                "status": "success",
                "message": "Identity Validated & Secured.",
                "distance": result["distance"],
            })
            session["verified"] = True
        else:
            await websocket.send_json({
                "status": "failed",
                "message": "Profile mismatch! Faces do not align.",
                "distance": result["distance"],
            })

        await websocket.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_session(session_id: str) -> None:
    """Remove the uploaded ID file and purge the session from memory."""
    session = sessions.pop(session_id, None)
    if session is None:
        return
    id_path = session.get("id_path", "")
    if id_path and os.path.exists(id_path):
        try:
            os.remove(id_path)
        except OSError as e:
            logger.warning("Could not delete %s: %s", id_path, e)
