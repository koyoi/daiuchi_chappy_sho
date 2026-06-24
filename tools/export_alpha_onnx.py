#!/usr/bin/env python3
"""Export ShogiResNetSE to ONNX format.

Usage:
  python tools/export_alpha_onnx.py --model alpha_model.pt --output alpha_model.onnx
  python tools/export_alpha_onnx.py --model alpha_model.pt --output alpha_model.onnx --verify
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Export ShogiResNetSE to ONNX")
    parser.add_argument("--model", default="alpha_model.pt", help="PyTorch model path")
    parser.add_argument("--output", default="alpha_model.onnx", help="ONNX output path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    import torch
    from torch import nn
    from train_alpha import build_model, INPUT_CHANNELS

    model = build_model(nn, channels=args.channels, num_blocks=args.blocks)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 1

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({params:,} params)", file=sys.stderr)

    dummy = torch.zeros(1, INPUT_CHANNELS, 9, 9)

    warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
    torch.onnx.export(
        model, dummy, args.output,
        dynamo=False, opset_version=args.opset,
        input_names=["features"],
        output_names=["value_wdl", "policy_logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "value_wdl": {0: "batch"},
            "policy_logits": {0: "batch"},
        },
    )
    print(f"Exported to {args.output}", file=sys.stderr)

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(args.output)

        test_batch = 4
        feat = torch.randn(test_batch, INPUT_CHANNELS, 9, 9)

        with torch.no_grad():
            pt_wdl, pt_policy = model(feat)

        feeds = {"features": feat.numpy()}
        onnx_wdl, onnx_policy = session.run(None, feeds)

        wdl_diff = np.max(np.abs(pt_wdl.numpy() - onnx_wdl))
        policy_diff = np.max(np.abs(pt_policy.numpy() - onnx_policy))
        print(f"Verification (batch={test_batch}):", file=sys.stderr)
        print(f"  WDL    max diff: {wdl_diff:.6e}", file=sys.stderr)
        print(f"  Policy max diff: {policy_diff:.6e}", file=sys.stderr)

        if wdl_diff < 1e-4 and policy_diff < 1e-4:
            print("  OK: outputs match", file=sys.stderr)
        else:
            print("  WARNING: outputs differ significantly", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
