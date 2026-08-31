"""Export a trained net to ONNX (and optionally int8), and verify it.

    python -m training.export --checkpoint weights/model.pt --out weights/model.onnx --quantize
    python -m training.export --random --config tiny --out weights/model.onnx   # plumbing test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from encoding import N_PLANES
from training.model import CONFIGS, ChessNet

OPSET = 18


def export_onnx(model: ChessNet, path: Path | str, quantize: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = next(model.parameters()).device
    model.eval().to("cpu")  # the exporter traces on CPU; a CUDA model trips it up
    try:
        dummy = torch.zeros(2, N_PLANES, 8, 8)
        batch = torch.export.Dim("batch")
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["planes"],
            output_names=["policy", "value"],
            dynamic_shapes={"x": {0: batch}},
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
        )
        _verify(model, path)
    finally:
        model.to(origin)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = path.with_suffix(".int8.onnx")
        quantize_dynamic(str(path), str(int8_path), weight_type=QuantType.QInt8)
        _verify(model, int8_path, tolerance=5e-2)
        print(f"quantised -> {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
    return path


def _verify(model: ChessNet, path: Path, tolerance: float = 1e-3) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((4, N_PLANES, 8, 8), dtype=np.float32)
    with torch.no_grad():
        ref_policy, ref_value = model(torch.from_numpy(sample))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    policy, value = session.run(None, {"planes": sample})
    policy_gap = float(np.abs(policy - ref_policy.numpy()).max())
    value_gap = float(np.abs(value - ref_value.numpy()).max())
    print(f"verify {path.name}: policy dmax {policy_gap:.2e}  value dmax {value_gap:.2e}")
    if policy_gap > tolerance or value_gap > tolerance:
        raise SystemExit(f"{path} disagrees with torch beyond {tolerance}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random", action="store_true", help="export an untrained net")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="medium")
    parser.add_argument("--out", type=Path, default=Path("weights/model.onnx"))
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu")
        model = ChessNet(CONFIGS[blob["config"]])
        model.load_state_dict(blob["state_dict"])
    elif args.random:
        model = ChessNet(CONFIGS[args.config])
    else:
        raise SystemExit("pass --checkpoint or --random")

    export_onnx(model, args.out, quantize=args.quantize)


if __name__ == "__main__":
    main()
