"""
MediaPipe dual-hand landmark extraction.

Strategy:
- Extract RIGHT hand (63 features) + LEFT hand (63 features) = 126 total
- If a hand is absent → zero vector for that hand
- Supports both single-hand datasets (WLASL) and dual-hand (LSA64)
- Normalize each hand relative to its own wrist (translation + scale invariant)

Config: set NUM_HANDS=1 in .env to use right-hand-only mode (63 features).
"""

import numpy as np
import mediapipe as mp
from loguru import logger

_mp_hands = mp.solutions.hands
_hands_instance = None

SINGLE_HAND_DIM = 21 * 3  # 63


def get_hands():
    global _hands_instance
    if _hands_instance is None:
        _hands_instance = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        logger.info("MediaPipe Hands initialized (dual-hand, model_complexity=1)")
    return _hands_instance


def extract_keypoints(frame_rgb: np.ndarray, num_hands: int = 2) -> np.ndarray:
    """
    Extract normalized keypoints from a single RGB frame.

    Args:
        frame_rgb:  H×W×3 uint8 RGB frame
        num_hands:  1 = right hand only (63-d), 2 = both hands (126-d)

    Returns:
        np.ndarray shape (63,) or (126,)
    """
    hands = get_hands()
    results = hands.process(frame_rgb)

    if num_hands == 1:
        return _extract_single(results)
    return _extract_dual(results)


def _extract_single(results) -> np.ndarray:
    """Right hand only → 63-d."""
    if not results.multi_hand_landmarks:
        return np.zeros(SINGLE_HAND_DIM, dtype=np.float32)

    target = None
    if results.multi_handedness:
        for i, h in enumerate(results.multi_handedness):
            if h.classification[0].label == "Right":
                target = results.multi_hand_landmarks[i]
                break
    if target is None:
        target = results.multi_hand_landmarks[0]

    return _normalize(target)


def _extract_dual(results) -> np.ndarray:
    """Both hands → 126-d (right 63 + left 63). Zero if absent."""
    right = np.zeros(SINGLE_HAND_DIM, dtype=np.float32)
    left = np.zeros(SINGLE_HAND_DIM, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return np.concatenate([right, left])

    for i, h in enumerate(results.multi_handedness):
        label = h.classification[0].label
        vec = _normalize(results.multi_hand_landmarks[i])
        if label == "Right":
            right = vec
        else:
            left = vec

    return np.concatenate([right, left])


def _normalize(hand_landmarks) -> np.ndarray:
    """Translate to wrist origin, scale by hand size."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    coords -= coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale
    return coords.flatten()


def close_hands():
    global _hands_instance
    if _hands_instance:
        _hands_instance.close()
        _hands_instance = None
