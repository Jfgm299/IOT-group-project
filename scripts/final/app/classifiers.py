from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class SklearnGestureClassifier:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return str(self.model.classes_[idx]), float(probs[idx])
