from __future__ import annotations

from pathlib import Path
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


def looks_like_probability(values: np.ndarray) -> bool:
    return bool(np.all(values >= 0.0) and np.all(values <= 1.0) and np.isclose(values.sum(), 1.0, atol=1e-3))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def main() -> None:
    root = Path(sys.argv[1])
    hef_path = sys.argv[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from gestures import ID_TO_LABEL

    hef = HEF(str(hef_path))
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()
    if len(input_infos) != 1 or len(output_infos) != 1:
        raise ValueError("The gesture HEF must have exactly one input and one output.")

    input_name = input_infos[0].name
    output_name = output_infos[0].name
    input_shape = tuple(input_infos[0].shape)
    expected_values = int(np.prod(input_shape))

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
                try:
                    features = np.asarray(json.loads(line), dtype=np.float32).reshape(-1)
                    if features.size != expected_values:
                        raise ValueError(
                            f"The HEF expects {expected_values} values {input_shape}, but got {features.size}."
                        )

                    sample = features.reshape((1, *input_shape))
                    result = infer_pipeline.infer({input_name: sample})
                    output = np.asarray(result[output_name]).reshape(-1).astype(np.float32)
                    probs = output if looks_like_probability(output) else softmax(output)
                    idx = int(np.argmax(probs))
                    print(json.dumps({"label": ID_TO_LABEL.get(idx, str(idx)), "conf": float(probs[idx])}), flush=True)
                except Exception as exc:
                    print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
