from __future__ import annotations

import numpy as np


def landmarks_to_feature_vector(landmarks: list[list[float]]) -> np.ndarray:
    """Convert 21 normalized hand landmarks into a translation/scale-stable 63-value vector."""
    arr = np.array(landmarks, dtype=np.float32)
    wrist = arr[0].copy()
    arr = arr - wrist

    scale = np.linalg.norm(arr[9])
    if scale < 1e-6:
        scale = 1.0
    arr = arr / scale

    return arr.flatten()
