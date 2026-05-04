from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def ensure_model_exists(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(MODEL_URL, model_path)
    return model_path


def draw_landmarks(frame, landmarks_norm: list[list[float]]) -> None:
    h, w = frame.shape[:2]
    points = []
    for x, y, _ in landmarks_norm:
        px, py = int(x * w), int(y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    for i, j in HAND_CONNECTIONS:
        if i < len(points) and j < len(points):
            cv2.line(frame, points[i], points[j], (255, 0, 0), 2)


class HandLandmarkerWrapper:
    def __init__(self, max_num_hands: int = 1, min_det: float = 0.5, min_track: float = 0.5, model_path: str | None = None):
        model_file = Path(model_path) if model_path else ensure_model_exists()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_file))
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_track,
            min_tracking_confidence=min_track,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect_one_hand_landmarks(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)
        if not results.hand_landmarks:
            return None
        return [[point.x, point.y, point.z] for point in results.hand_landmarks[0]]

    def close(self) -> None:
        self.landmarker.close()
