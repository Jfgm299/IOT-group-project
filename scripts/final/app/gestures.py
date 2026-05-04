GESTURE_LABELS = [
    "rock",
    "paper",
    "scissors",
    "lizard",
    "rocknroll",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(GESTURE_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
