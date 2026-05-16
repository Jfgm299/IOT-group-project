from __future__ import annotations

from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import requests

APP_DIR = Path(__file__).resolve().parent
FINAL_DIR = APP_DIR.parent
WORKER_DIR = FINAL_DIR / "workers"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import cv2
import numpy as np

from camera_sources import create_camera_source
from classifiers import SklearnGestureClassifier
from features import landmarks_to_feature_vector
from mediapipe_utils import HandLandmarkerWrapper, draw_landmarks


# Game's logic
from game import match
# Bluetooth communication to Pico
from pico_ble import send_data

def stable_vote(values: deque[str]) -> str:
    if not values:
        return "-"
    return Counter(values).most_common(1)[0][0]


def hailo_runtime_available(hailo_python: str) -> bool:
    return subprocess.run(
        [hailo_python, "-c", "import hailo_platform"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


class JsonWorker:
    def __init__(self, command: list[str], name: str) -> None:
        self.name = name
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.read_response()
        if not ready.get("ready"):
            raise RuntimeError(f"{name} worker did not start correctly: {ready}")
        self.ready = ready

    def request(self, payload) -> dict:
        if self.proc.stdin is None:
            raise RuntimeError(f"{self.name} worker stdin is not available")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        response = self.read_response()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def read_response(self) -> dict:
        if self.proc.stdout is None:
            raise RuntimeError(f"{self.name} worker stdout is not available")
        line = self.proc.stdout.readline()
        if not line:
            stderr = ""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read()
            raise RuntimeError(f"{self.name} worker exited unexpectedly. {stderr}".strip())
        return json.loads(line)

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class HailoGestureClassifier:
    def __init__(self, hef_path: str, hailo_python: str) -> None:
        worker = WORKER_DIR / "hailo_gesture_worker.py"
        self.worker = JsonWorker([hailo_python, "-u", str(worker), str(APP_DIR), hef_path], "Hailo gesture")

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        response = self.worker.request(features.reshape(-1).astype(float).tolist())
        return str(response["label"]), float(response["conf"])

    def close(self) -> None:
        self.worker.close()


class HailoHandLandmarker:
    def __init__(self, hef_path: str, hailo_python: str) -> None:
        worker = WORKER_DIR / "hailo_hand_worker.py"
        self.worker = JsonWorker([hailo_python, "-u", str(worker), str(APP_DIR), hef_path], "Hailo hand landmark")
        shape = self.worker.ready.get("shape") or [224, 224, 3]
        self.input_h = int(shape[0])
        self.input_w = int(shape[1])

    def detect(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[list[list[float]] | None, float]:
        x0, y0, x1, y1 = clamp_roi(roi, frame.shape[1], frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None, 0.0

        crop = frame[y0:y1, x0:x1]
        resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_b64 = base64.b64encode(rgb.tobytes()).decode("ascii")
        response = self.worker.request({"image": image_b64})

        crop_w = x1 - x0
        crop_h = y1 - y0
        landmarks = []
        for lx, ly, lz in response["landmarks"]:
            fx = (x0 + float(lx) * crop_w) / frame.shape[1]
            fy = (y0 + float(ly) * crop_h) / frame.shape[0]
            fz = float(lz) * crop_w / frame.shape[1]
            landmarks.append([fx, fy, fz])
        return landmarks, float(response.get("confidence", 1.0))

    def close(self) -> None:
        self.worker.close()


class HybridHandLandmarker:
    def __init__(self, hef_path: str, hailo_python: str, refresh_frames: int, roi_padding: float) -> None:
        self.hailo = HailoHandLandmarker(hef_path, hailo_python)
        self.fallback = HandLandmarkerWrapper(max_num_hands=1)
        self.refresh_frames = max(1, refresh_frames)
        self.roi_padding = roi_padding
        self.roi: tuple[int, int, int, int] | None = None
        self.frames_since_refresh = self.refresh_frames

    def detect_one_hand_landmarks(self, frame: np.ndarray) -> list[list[float]] | None:
        needs_refresh = self.roi is None or self.frames_since_refresh >= self.refresh_frames
        if needs_refresh:
            landmarks = self.fallback.detect_one_hand_landmarks(frame)
            self.frames_since_refresh = 0
            if landmarks is None:
                self.roi = None
                return None
            self.roi = landmarks_to_roi(landmarks, frame.shape[1], frame.shape[0], self.roi_padding)
            return landmarks

        self.frames_since_refresh += 1
        assert self.roi is not None
        landmarks, confidence = self.hailo.detect(frame, self.roi)
        if landmarks is None or confidence < 0.4:
            self.roi = None
            return None
        self.roi = landmarks_to_roi(landmarks, frame.shape[1], frame.shape[0], self.roi_padding)
        return landmarks

    def close(self) -> None:
        self.hailo.close()
        self.fallback.close()


def landmarks_to_roi(landmarks: list[list[float]], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    xs = [point[0] * width for point in landmarks]
    ys = [point[1] * height for point in landmarks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    side = max(x1 - x0, y1 - y0, 32.0) * (1.0 + padding)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return clamp_roi((int(cx - side / 2.0), int(cy - side / 2.0), int(cx + side / 2.0), int(cy + side / 2.0)), width, height)


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


class InferenceState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status = "Starting..."
        self.current_label = "-" # < -- (Mateo) to read dectentios
        self.streak = 0 # < -- (Mateo) to save streak in backend
        self.last_score = 0 # < -- (Mateo) to send streak to database
        self.send_status = True # < -- (Mateo) to verify that the data was sent to the pico
        self.stopped = False


def resolve_gesture_backend(args: argparse.Namespace) -> tuple[str, str | None, str]:
    hef = args.hef
    default_hefs = [FINAL_DIR / "models" / "hailo" / "gesture_model.hef", FINAL_DIR / "models" / "gesture_model.hef"]
    if hef is None:
        for default_hef in default_hefs:
            if default_hef.exists():
                hef = str(default_hef)
                break

    if args.backend == "auto":
        if hef is not None and hailo_runtime_available(args.hailo_python):
            return "hailo", hef, f"Gesture Hailo: {hef}"
        return "sklearn", None, "Gesture sklearn/CPU"

    if args.backend == "hailo":
        if hef is None:
            raise ValueError("--backend hailo requires --hef")
        return "hailo", hef, f"Gesture Hailo: {hef}"

    return "sklearn", None, "Gesture sklearn/CPU"


def resolve_landmark_backend(args: argparse.Namespace) -> tuple[str, str | None, str]:
    hef = args.landmark_hef
    default_hef = FINAL_DIR / "models" / "hailo" / "hand_landmark_lite.hef"
    if hef is None and default_hef.exists():
        hef = str(default_hef)

    if args.landmark_backend == "auto":
        if hef is not None and hailo_runtime_available(args.hailo_python):
            return "hailo", hef, f"Landmarks Hailo: {hef}"
        return "mediapipe", None, "Landmarks MediaPipe/CPU"

    if args.landmark_backend == "hailo":
        if hef is None:
            raise ValueError("--landmark-backend hailo requires --landmark-hef")
        return "hailo", hef, f"Landmarks Hailo: {hef}"

    return "mediapipe", None, "Landmarks MediaPipe/CPU"


def inference_loop(args: argparse.Namespace, state: InferenceState) -> None:
    gesture_backend, gesture_hef, gesture_status = resolve_gesture_backend(args)
    landmark_backend, landmark_hef, landmark_status = resolve_landmark_backend(args)
    status = f"{gesture_status} | {landmark_status}"
    with state.lock:
        state.status = status
    print(status, flush=True)

    camera = create_camera_source(args.camera_source, args.camera, args.width, args.height)
    detector = (
        HybridHandLandmarker(str(landmark_hef), args.hailo_python, args.landmark_refresh, args.roi_padding)
        if landmark_backend == "hailo"
        else HandLandmarkerWrapper(max_num_hands=1)
    )
    classifier = HailoGestureClassifier(str(gesture_hef), args.hailo_python) if gesture_backend == "hailo" else SklearnGestureClassifier(args.model)
    last_preds: deque[str] = deque(maxlen=args.window)

    frame_interval = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    last_frame_at = 0.0
    frame_count = 0
    label = "-"
    conf = 0.0

    try:
        while not state.stopped:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if frame_interval and now - last_frame_at < frame_interval:
                continue
            last_frame_at = now

            frame_count += 1
            if frame_count % args.infer_every == 0:
                detection = detector.detect_one_hand_landmarks(frame)
                label = "-"
                conf = 0.0
                if detection is not None:
                    draw_landmarks(frame, detection)
                    features = landmarks_to_feature_vector(detection)
                    raw_label, conf = classifier.predict(features)
                    last_preds.append(raw_label if conf >= args.threshold else "-")
                    label = stable_vote(last_preds)
                else:
                    last_preds.append("-")
            else:
                label = stable_vote(last_preds)

            cv2.putText(frame, f"Pred: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if encoded:
                with state.lock:
                    state.jpeg = buffer.tobytes()
                    state.status = status
                    state.current_label = label # <-- (Mateo)
    finally:
        detector.close()
        close_classifier = getattr(classifier, "close", None)
        if close_classifier is not None:
            close_classifier()
        camera.release()


def make_handler(state: InferenceState): # Changes by Mateo: Added handler to send the current detection back to the script to play the game
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None: 
            if self.path in {"/", "/index.html"}:
                with state.lock:
                    status = state.status
                body = f"""
<!doctype html>
<html>
<head>
<title>Gesture Inference</title>
<style>
  /* Simple spinning loading ring */
  .loader {{
      border: 4px solid #333;
      border-top: 4px solid #00ff00;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: spin 1s linear infinite;
      display: inline-block;
      vertical-align: middle;
      margin-right: 10px;
    }}
    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    .loading-text {{
      color: #ffaa00 !important;
      opacity: 0.7;
    }}
    button:disabled {{
  opacity: 0.5;
  cursor: not-allowed;
    }}
</style>
</head>
<body style="margin:0;background:#111;colorstatus:white;font-family:sans-serif;text-align:center">
  <h2>Gesture Inference</h2>
  <p style="margin:0 0 12px 0;color:#ccc">{status}</p>
  <img src="/stream.mjpg" style="max-width:100vw;max-height:82vh" />
  <div id="detection-log" style="margin-top:20px; font-size: 24px; color: #00ff00; font-weight: bold;">
    Press Space to play match
  </div>
  <div id="bluetooth-status" style="margin-top:20px; font-size: 24px; color: #00ff00; font-weight: bold;">
    Send status to the pico: -
  </div>
  <div id="save-score-area" style="display:none; margin-top: 20px;">
    <input type="text" id="player-code" maxlength="3" placeholder="AAA" style="text-transform:uppercase; font-size: 20px; width: 60px; text-align: center;">
    <button onclick="sendCode()" style="font-size: 20px; cursor: pointer;">Send</button>
  </div>

  <script>
    let lastStreak = 0;
    let isProcessing = false; // Flag for spacebar/detections
    let isSavingCode = false; // Flag for database submissions

    function sendCode(){{
        
        if(isSavingCode) return;    

        const codeInput = document.getElementById('player-code')
        const sendButton = codeInput.nextElementSibling;
        const code = codeInput.value;

        if (code.length !== 3) {{
            alert("Code must be 3 characters!");
            return;
        }}

        isSavingCode = true;
        sendButton.disabled = true;
        const originalButtonText = sendButton.innerText;
        sendButton.innerText = "Saving...";

        fetch('/save_code?code=' + encodeURIComponent(code))
            .then(response => {{
                // Manually force HTTP errors (like 500) to jump to the .catch() block
                if (!response.ok) {{
                    throw new Error("Server responded with an error status: " + response.status);
                }}
                return response.json();
            }})
            .then(() => {{
                document.getElementById('save-score-area').style.display = 'none';
                document.getElementById('player-code').value = '';
                alert("Code saved!");
            }})
            .catch(err => {{
                alert("Failed to save code to server.");
                console.error(err);
            }})
            .finally(() => {{
                // ALWAYS unlock the state and restore button text
                isSavingCode = false;
                sendButton.disabled = false;
                sendButton.innerText = originalButtonText;
            }});;

    }}

    // Note the double curly braces below for the f-string to ignore them
    document.addEventListener('keydown', function(event) {{
      if (event.code === 'Space') {{
        event.preventDefault(); // Stop page from scrolling

        //IF ALREADY LOADING, DO NOTHING AND RETURN
        if(isProcessing){{
            return;
        }}

        isProcessing = true; // Set flag to true to lock the spacebar

        const logDiv = document.getElementById('detection-log');
        logDiv.innerHTML = '<div class="loader"></div> Calculating match...';
        logDiv.className = "loading-text";

        fetch('/current_detection')
          .then(response => response.json())
          .then(data => {{
          
            // Restore to normal
            logDiv.className = "";
            logDiv.style.color = "#00ff00";
            logDiv.innerText = data.display_text + " | Streak: " + data.streak;

            const statusDisplay = document.getElementById('bluetooth-status');
            statusDisplay.innerText = "Send status to the pico: " + data.send_status;  
                    
            if (data.last_score > 0) {{
                document.getElementById('save-score-area').style.display = 'block';
            }}
          }})
          .catch(err => {{
            logDiv.style.color = "#ff0000";
            logDiv.innerText = "Error fetching detection.";
            console.error(err);
          }})
          .finally(() => {{
            // ALWAYS unlock the spacebar when the request finishes (success or failure)
            isProcessing = false;
          }});
      }}
    }});
  </script>
</body>
</html>
""".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            
            if self.path.startswith("/save_code"):
                # Simple way to parse the ?code=AAA part of the URL
                from urllib.parse import urlparse, parse_qs
                query_components = parse_qs(urlparse(self.path).query)
                code = query_components.get("code", ["???"])[0]

                with state.lock:
                    final_score = state.last_score
                    state.last_score = 0 # Reset streak after saving
                
                print(f"PLAYER LOGGED CODE: [{code}] with streak: {final_score}") # This prints in your terminal

                database_api_url = "http://mateodonado.local:8000/statistics"
                payload = {
                    "username": code,
                    "streak": final_score
                }

                try:
                    # We send a POST request to your database backend
                    # Using a timeout=3 ensures your local camera feed doesn't freeze if the DB is slow
                    response = requests.post(database_api_url, json=payload, timeout=3)
        
                    if response.status_code == 201 or response.status_code == 200:
                        print(f"Successfully synced {code}'s score to database.")
                    else:
                        print(f"Database rejected score: {response.status_code}")
                except Exception as e:
                    print(f"Failed to connect to database backend: {e}")
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"NOK")
                    return


                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
                return
            
            if self.path == "/current_detection": # (Mateo)
                with state.lock:
                    current = state.current_label
                
                if current != "-":
                    try:
                        game_result, cpu_choice = match(current, cpu_level= 1) # Get the game result from game.py

                        # --- BRIDGE SYNC TO ASYNC HERE ---
                        import asyncio
                        try:
                            # We use asyncio.run to execute the async function inside this sync block
                            send_result = asyncio.run(send_data(cpu_choice))
                        except Exception as ble_err:
                            print(f"Bluetooth transmission failed: {ble_err}")
                            send_result = False
                        # ----------------------------------

                        if send_result:
                            state.send_status = True
                        else:
                            state.send_status = False

                        if game_result == 'Player':
                            state.streak += 1
                        elif game_result == 'AI':

                            state.last_score = state.streak # Save streak to send to database
                            state.streak = 0
                        
                        display_text = f"Gesture: {current} | Result: {game_result}"
                    except Exception as e:
                        display_text = f"Error in match {str(e)}"
                        game_result = "Error"
                else:
                    display_text = "No hand detected"
                    game_result = "-"
                
                streak = state.streak
                send_status = state.send_status
                    
    
                response_data = json.dumps({"gesture": current,
                                             "result": game_result,
                                             "display_text": display_text,
                                             "streak": streak,
                                             "send_status": send_status,
                                             "last_score": state.last_score }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_jpeg: bytes | None = None
            while not state.stopped:
                with state.lock:
                    jpeg = state.jpeg
                if jpeg is None or jpeg is last_jpeg:
                    time.sleep(0.01)
                    continue
                last_jpeg = jpeg
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser-based gesture inference with optional Hailo acceleration.")
    parser.add_argument("--model", default=str(FINAL_DIR / "models" / "gesture_model.joblib"))
    parser.add_argument("--backend", choices=["auto", "sklearn", "hailo"], default="auto")
    parser.add_argument("--hef", default=None, help="Path to a gesture classifier HEF compatible with 63 landmark features.")
    parser.add_argument("--landmark-backend", choices=["auto", "mediapipe", "hailo"], default="auto")
    parser.add_argument("--landmark-hef", default=None, help="Path to hand_landmark_lite.hef for Hailo landmark acceleration.")
    parser.add_argument("--landmark-refresh", type=int, default=10, help="Frames between MediaPipe ROI refreshes when landmarks use Hailo.")
    parser.add_argument("--roi-padding", type=float, default=1.0, help="Padding around the hand crop used by the Hailo landmark model.")
    parser.add_argument("--hailo-python", default="python3", help="System Python that can import hailo_platform.")
    parser.add_argument("--camera-source", choices=["auto", "picamera2", "rpicam", "opencv"], default="opencv")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--max-fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.infer_every < 1:
        raise ValueError("--infer-every must be >= 1")
    if args.landmark_refresh < 1:
        raise ValueError("--landmark-refresh must be >= 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    state = InferenceState()
    worker = threading.Thread(target=inference_loop, args=(args, state), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Open http://<PI_IP>:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        state.stopped = True
        server.server_close()


if __name__ == "__main__":
    main()
