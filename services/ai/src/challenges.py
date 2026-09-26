"""
challenges.py — Active Liveness Challenge Detectors

Provides two challenge functions that operate on MediaPipe Face Landmarker
output (a list of NormalizedLandmark objects):

    detect_smile(landmarks, w, h)   → bool
    detect_head_yaw(landmarks, w, h) → "LEFT" | "RIGHT" | None

Both rely on the same 478-landmark mesh already extracted by liveness.py,
so there is zero duplicate inference cost.
"""

import math

import cv2
import numpy as np

# =========================================================================
# Landmark indices (MediaPipe Face Mesh 478-point topology)
# =========================================================================
# Mouth corners
LM_MOUTH_LEFT = 61      # Left corner of the mouth
LM_MOUTH_RIGHT = 291    # Right corner of the mouth

# Upper / lower lip midpoints (for mouth openness, optional future use)
LM_UPPER_LIP = 13
LM_LOWER_LIP = 14

# Face width reference points (outer cheek / jaw area)
LM_FACE_LEFT = 234
LM_FACE_RIGHT = 454

# Head pose estimation anchors
LM_NOSE_TIP = 1
LM_CHIN = 152
LM_LEFT_EYE_CORNER = 33
LM_RIGHT_EYE_CORNER = 263
LM_LEFT_MOUTH_CORNER = 61
LM_RIGHT_MOUTH_CORNER = 291

# =========================================================================
# Thresholds (tuned empirically)
# =========================================================================
SMILE_RATIO_THRESH = 0.43    # mouth_width / face_width  (neutral ≈ 0.35, smile ≈ 0.45+)
YAW_ANGLE_THRESH = 12.0      # degrees — moderate turn (partial profile) is enough


# =========================================================================
# Helper
# =========================================================================

def _px(landmark, w, h):
    """Convert a normalised landmark to (x, y) pixel coordinates."""
    return (landmark.x * w, landmark.y * h)


# =========================================================================
# 1. Smile Detection
# =========================================================================

def detect_smile(landmarks, w: int, h: int) -> bool:
    """
    Detect a smile by comparing mouth width to face width.

    Math
    ----
    mouth_width  = ‖landmark[61] − landmark[291]‖   (Euclidean, pixels)
    face_width   = ‖landmark[234] − landmark[454]‖   (Euclidean, pixels)
    smile_ratio  = mouth_width / face_width

    When the user smiles, the mouth corners spread outward, increasing
    the ratio from ~0.35 (neutral) to ~0.45+ (smile).

    Args:
        landmarks: List of NormalizedLandmark from FaceLandmarker.
        w, h:      Frame width and height in pixels.

    Returns:
        True if a clear smile is detected.
    """
    mouth_left = _px(landmarks[LM_MOUTH_LEFT], w, h)
    mouth_right = _px(landmarks[LM_MOUTH_RIGHT], w, h)
    face_left = _px(landmarks[LM_FACE_LEFT], w, h)
    face_right = _px(landmarks[LM_FACE_RIGHT], w, h)

    mouth_width = math.dist(mouth_left, mouth_right)
    face_width = math.dist(face_left, face_right)

    if face_width == 0:
        return False

    smile_ratio = mouth_width / face_width
    return smile_ratio >= SMILE_RATIO_THRESH


# =========================================================================
# 2. Head Yaw (Turn Left / Right) via cv2.solvePnP
# =========================================================================

# Generic 3D model points for a canonical face (in arbitrary units).
# These are approximate positions of the landmarks in a "standard" face,
# centred at the nose tip.  Units don't matter — only ratios count.
_3D_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # Nose tip
    (0.0,   -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0,  170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0,  -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

# Corresponding landmark indices (same order as _3D_MODEL_POINTS)
_2D_LANDMARK_IDS = [
    LM_NOSE_TIP,
    LM_CHIN,
    LM_LEFT_EYE_CORNER,
    LM_RIGHT_EYE_CORNER,
    LM_LEFT_MOUTH_CORNER,
    LM_RIGHT_MOUTH_CORNER,
]


def detect_head_yaw(landmarks, w: int, h: int) -> str | None:
    """
    Estimate head yaw angle and classify as LEFT, RIGHT, or None (neutral).

    Math (cv2.solvePnP)
    -------------------
    Given 6 known 3D model points and their corresponding 2D projections
    in the image, solvePnP recovers the rotation vector (Rodrigues form).
    We convert this to a full rotation matrix, then decompose it into
    Euler angles.  The Y-axis rotation = yaw (horizontal head turn).

    Positive yaw → user looks to their LEFT  (camera's right)
    Negative yaw → user looks to their RIGHT (camera's left)

    Args:
        landmarks: List of NormalizedLandmark from FaceLandmarker.
        w, h:      Frame width and height in pixels.

    Returns:
        "LEFT"  if yaw > +YAW_ANGLE_THRESH,
        "RIGHT" if yaw < −YAW_ANGLE_THRESH,
        None    if the head is roughly centred.
    """
    # Build the 2D projection array from the landmarks
    image_points = np.array(
        [_px(landmarks[i], w, h) for i in _2D_LANDMARK_IDS],
        dtype=np.float64,
    )

    # Approximate camera intrinsic matrix (no lens distortion)
    focal_length = w
    centre = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0,            centre[0]],
        [0,            focal_length, centre[1]],
        [0,            0,            1.0      ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    success, rotation_vec, _translation_vec = cv2.solvePnP(
        _3D_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # Convert Rodrigues rotation vector → 3×3 rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose into Euler angles (in degrees)
    # proj_matrix is [R | t] but decomposeProjectionMatrix needs 3×4
    proj_matrix = np.hstack((rotation_mat, _translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    yaw = euler_angles[1][0]  # Y-axis rotation in degrees

    if yaw > YAW_ANGLE_THRESH:
        return "LEFT"
    elif yaw < -YAW_ANGLE_THRESH:
        return "RIGHT"
    return None
