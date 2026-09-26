"""
face_match.py — Face Verification via DeepFace

Compares an uploaded ID photo against a live webcam frame using
the VGG-Face model. Returns a structured result dict indicating
match/mismatch and a normalised similarity score.
"""

def compare_faces(id_img_path: str, live_frame) -> dict:
    """
    Compare the uploaded ID picture with a live captured frame.

    Args:
        id_img_path: Filesystem path to the pre-uploaded ID photo.
        live_frame:  BGR numpy array captured from the webcam.

    Returns:
        dict with keys:
            success  (bool)  — whether the comparison executed without error.
            verified (bool)  — True if faces match (only when success=True).
            distance (float) — raw dissimilarity score from DeepFace.
            similarity (float) — approximate 0-1 similarity percentage.
            error    (str)   — human-readable message (only when success=False).
    """
    try:
        from deepface import DeepFace

        result = DeepFace.verify(
            img1_path=id_img_path,
            img2_path=live_frame,
            model_name="VGG-Face",
            enforce_detection=True,
        )

        distance = result.get("distance", 0.0)

        return {
            "success": True,
            "verified": result.get("verified", False),
            "distance": distance,
            "similarity": max(0.0, 1.0 - distance),
        }

    except ValueError:
        # DeepFace raises ValueError when no face region is detected
        return {
            "success": False,
            "error": "No face detected in the live frame. Please reposition.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"DeepFace error: {e}",
        }
