#!/usr/bin/env python3
"""Export ShogiTransformer to ONNX format.

Usage:
  python tools/export_onnx.py --model nn_model.pt --output nn_model.onnx
  python tools/export_onnx.py --model nn_model.pt --output nn_model.onnx --verify
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Export ShogiTransformer to ONNX")
    parser.add_argument("--model", default="nn_model.pt", help="PyTorch model path")
    parser.add_argument("--output", default="nn_model.onnx", help="ONNX output path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    import torch
    from torch import nn
    from train import build_model

    model = build_model(nn)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 1

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({params:,} params)", file=sys.stderr)

    batch_size = 1
    board_tokens = torch.zeros(batch_size, 81, dtype=torch.long)
    hand_tokens = torch.zeros(batch_size, 14, dtype=torch.long)
    side_token = torch.zeros(batch_size, dtype=torch.long)
    own_atk = torch.zeros(batch_size, 81, dtype=torch.long)
    opp_atk = torch.zeros(batch_size, 81, dtype=torch.long)

    torch.onnx.export(
        model,
        (board_tokens, hand_tokens, side_token, own_atk, opp_atk),
        args.output,
        dynamo=False,
        opset_version=args.opset,
        input_names=["board_tokens", "hand_tokens", "side_token", "own_atk", "opp_atk"],
        output_names=["value", "policy_logits"],
        dynamic_axes={
            "board_tokens": {0: "batch"},
            "hand_tokens": {0: "batch"},
            "side_token": {0: "batch"},
            "own_atk": {0: "batch"},
            "opp_atk": {0: "batch"},
            "value": {0: "batch"},
            "policy_logits": {0: "batch"},
        },
    )
    print(f"Exported to {args.output}", file=sys.stderr)

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(args.output)

        test_batch = 4
        bt = torch.randint(0, 29, (test_batch, 81))
        ht = torch.randint(0, 19, (test_batch, 14))
        st = torch.randint(0, 2, (test_batch,))
        oa = torch.randint(0, 9, (test_batch, 81))
        op = torch.randint(0, 9, (test_batch, 81))

        with torch.no_grad():
            pt_value, pt_policy = model(bt, ht, st, oa, op)

        feeds = {
            "board_tokens": bt.numpy(),
            "hand_tokens": ht.numpy(),
            "side_token": st.numpy(),
            "own_atk": oa.numpy(),
            "opp_atk": op.numpy(),
        }
        onnx_value, onnx_policy = session.run(None, feeds)

        value_diff = np.max(np.abs(pt_value.numpy() - onnx_value))
        policy_diff = np.max(np.abs(pt_policy.numpy() - onnx_policy))
        print(f"Verification (batch={test_batch}):", file=sys.stderr)
        print(f"  Value  max diff: {value_diff:.6e}", file=sys.stderr)
        print(f"  Policy max diff: {policy_diff:.6e}", file=sys.stderr)

        if value_diff < 1e-4 and policy_diff < 1e-4:
            print("  OK: outputs match", file=sys.stderr)
        else:
            print("  WARNING: outputs differ significantly", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
