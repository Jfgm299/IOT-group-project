from __future__ import annotations

from pathlib import Path
import base64
import json
import sys

import numpy as np
from hailo_platform import (  # type: ignore
    ConfigureParams,
    FormatType,
    HEF,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def score_output(arrays: list[np.ndarray]) -> float:
    scores = []
    for values in arrays:
        flat = values.reshape(-1).astype(np.float32)
        if flat.size == 1:
            value = float(flat[0])
            scores.append(value if 0.0 <= value <= 1.0 else sigmoid(value))
    return max(scores) if scores else 1.0


def choose_landmarks(arrays: list[np.ndarray]) -> np.ndarray:
    candidates = [values.reshape(-1).astype(np.float32) for values in arrays if values.size == 63]
    if not candidates:
        raise ValueError("The HEF did not return a 63-value landmark output.")

    def candidate_score(values: np.ndarray) -> tuple[bool, float]:
        points = values.reshape(21, 3)
        x_span = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        y_span = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        max_xy = float(np.max(np.abs(points[:, :2])))
        return max_xy > 2.0, x_span + y_span

    return max(candidates, key=candidate_score)


def main() -> None:
    root = Path(sys.argv[1])
    hef_path = sys.argv[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    hef = HEF(str(hef_path))
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_infos = hef.get_input_vstream_infos()
    if len(input_infos) != 1:
        raise ValueError("The hand landmark HEF must have exactly one input.")

    input_name = input_infos[0].name
    input_shape = tuple(input_infos[0].shape)
    expected_values = int(np.prod(input_shape))

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print(json.dumps({"ready": True, "shape": input_shape}), flush=True)
            for line in sys.stdin:
                try:
                    payload = json.loads(line)
                    raw = base64.b64decode(payload["image"])
                    sample = np.frombuffer(raw, dtype=np.uint8).copy()
                    if sample.size != expected_values:
                        raise ValueError(
                            f"The HEF expects {expected_values} bytes {input_shape}, but got {sample.size}."
                        )

                    sample = sample.reshape((1, *input_shape))
                    result = infer_pipeline.infer({input_name: sample})
                    arrays = [np.asarray(values) for values in result.values()]
                    landmarks = choose_landmarks(arrays).reshape(21, 3).copy()
                    confidence = score_output(arrays)

                    # MediaPipe hand_landmark_lite usually returns x/y in 224x224 crop pixels.
                    max_xy = float(np.max(np.abs(landmarks[:, :2])))
                    if max_xy > 2.0:
                        landmarks[:, 0] /= float(input_shape[1])
                        landmarks[:, 1] /= float(input_shape[0])
                        landmarks[:, 2] /= float(input_shape[1])

                    print(json.dumps({"landmarks": landmarks.tolist(), "confidence": confidence}), flush=True)
                except Exception as exc:
                    print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
