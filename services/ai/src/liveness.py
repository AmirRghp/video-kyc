"""
liveness.py — Face Landmark Extraction & Blink Detection (EAR)

Provides:
    detect_landmarks(frame)  → list[NormalizedLandmark] | None
    detect_ear(frame)        → float | None

Uses MediaPipe Tasks API (FaceLandmarker) to extract 478 facial landmarks.
The shared `detect_landmarks()` function is also consumed by challenges.py
for smile and head-yaw detection — no duplicate model inference.
"""

import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker (shared singleton)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

_base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    output_face_blendshapes=False,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(_options)

# Landmark indices for the 6-point eye contour used in EAR calculation
# Order: [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_landmarks(frame):
    """
    Run FaceLandmarker on a BGR frame and return the first face's landmarks.

    Args:
        frame: BGR numpy array (OpenCV format).

    Returns:
        list[NormalizedLandmark] — 478 landmarks for the detected face, or
        None                    — if no face is found.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def detect_ear(frame):
    """
    Detect a face and return the averaged Eye Aspect Ratio (EAR).

    EAR formula:  (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

    Args:
        frame: BGR numpy array from OpenCV.

    Returns:
        float  — averaged EAR of both eyes, or
        None   — if no face is detected.
    """
    landmarks = detect_landmarks(frame)
    if landmarks is None:
        return None

    h, w, _ = frame.shape

    left_pts = [_px(landmarks, i, w, h) for i in LEFT_EYE]
    right_pts = [_px(landmarks, i, w, h) for i in RIGHT_EYE]

    left_ear = _ear(left_pts)
    right_ear = _ear(right_pts)

    return (left_ear + right_ear) / 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _px(landmark_list, index, w, h):
    """Convert a normalised landmark to [x, y] pixel coordinates."""
    lm = landmark_list[index]
    return [lm.x * w, lm.y * h]


def _ear(eye_points):
    """Compute the Eye Aspect Ratio for a 6-point eye contour."""
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    h = math.dist(eye_points[0], eye_points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)
