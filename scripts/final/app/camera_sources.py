from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Protocol

import cv2
import numpy as np


class CameraSource(Protocol):
    def read(self): ...

    def release(self) -> None: ...


@dataclass
class OpenCVCameraSource:
    camera_index: int = 0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


@dataclass
class PiCamera2Source:
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("picamera2 is not installed. On Raspberry Pi OS: sudo apt install -y python3-picamera2") from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        rgb_frame = self.picam2.capture_array()
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return True, bgr_frame

    def release(self) -> None:
        self.picam2.stop()


@dataclass
class RpicamMjpegSource:
    """Camera Module source using rpicam-vid MJPEG stdout."""

    width: int = 640
    height: int = 480
    framerate: int = 30

    def __post_init__(self) -> None:
        cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "--inline",
            "--codec",
            "mjpeg",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.framerate),
            "-o",
            "-",
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except FileNotFoundError as exc:
            raise RuntimeError("rpicam-vid was not found. Install it with: sudo apt install -y rpicam-apps") from exc

        self.buffer = bytearray()

    def read(self):
        if self.proc.stdout is None:
            return False, None

        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)

            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                jpg = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame

    def release(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def create_camera_source(source: str, camera_index: int = 0, width: int = 640, height: int = 480) -> CameraSource:
    if source == "rpicam":
        return RpicamMjpegSource(width=width, height=height)
    if source == "picamera2":
        return PiCamera2Source(width=width, height=height)
    if source == "opencv":
        return OpenCVCameraSource(camera_index=camera_index, width=width, height=height)
    if source == "auto":
        try:
            return PiCamera2Source(width=width, height=height)
        except Exception:
            return OpenCVCameraSource(camera_index=camera_index, width=width, height=height)
    raise ValueError(f"Unsupported camera source: {source}")
