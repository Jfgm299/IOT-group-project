# Gesture Web Inference

This folder contains a self-contained web inference pipeline for Raspberry Pi + Hailo AI HAT.

## Files

- `main.py`: top-level entry point.
- `app/infer_web.py`: web server and inference loop.
- `app/camera_sources.py`: Raspberry Pi/OpenCV camera sources.
- `app/mediapipe_utils.py`: MediaPipe fallback for ROI refresh and CPU landmarks.
- `app/features.py`: converts hand landmarks into the 63-value classifier input.
- `app/classifiers.py`: loads the sklearn `gesture_model.joblib` classifier.
- `app/gestures.py`: label definitions.
- `app/game.py`: game rules and match logic
- `workers/hailo_hand_worker.py`: Hailo worker for `hand_landmark_lite.hef`.
- `workers/hailo_gesture_worker.py`: optional Hailo worker for a future gesture-classifier HEF.
- `models/gesture_model.joblib`: trained gesture classifier.
- `models/hand_landmarker.task`: MediaPipe model used for CPU fallback and ROI refresh.
- `models/hailo/hand_landmark_lite.hef`: Hailo-8L hand landmark model.

## Python Dependencies

Install the Python dependencies in your virtual environment:

```bash
python -m pip install -r requirements.txt
```

On Raspberry Pi OS, OpenCV/MediaPipe setups often work better with the existing project venv or system packages. The Hailo Python API is installed by the Raspberry Pi Hailo packages, not by `requirements.txt`.

## System Dependencies

Install HailoRT and camera tools on the Raspberry Pi:

```bash
sudo apt install -y hailo-all rpicam-apps
```

The Hailo worker is launched with `python3` by default because the system Python can import `hailo_platform`.

## Run

From this folder:

```bash
python main.py
```

Or from the repository root:

```bash
python scripts/final/main.py
```

Open the printed URL in your browser:

```text
http://<PI_IP>:8080
```

## Explicit Hailo Landmark Mode

```bash
python main.py \
  --landmark-backend hailo \
  --landmark-hef models/hailo/hand_landmark_lite.hef \
  --model models/gesture_model.joblib
```

## Notes

- Hailo accelerates the hand landmark model.
- The gesture classifier is currently `gesture_model.joblib`, which runs on CPU and is lightweight.
- `hailo_gesture_worker.py` is included for future use if a 63-feature gesture classifier is compiled to HEF.
- Do not commit virtual environments such as `.venv`, `.venv311`, or `venv`.
- (Mateo) To make it work in laptop, change  `infer_web.py` line 497 default to opencv
