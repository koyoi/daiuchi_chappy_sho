#!/usr/bin/env python3
"""Export a PyTorch MLP model (mlp_model.pt) to a text weights file (mlp.weights).

Fuses the leading BatchNorm into the first Linear layer so the C++ side
only needs plain Linear+LeakyReLU layers.

Format:
  Line 1: input_dim hidden1_dim hidden2_dim
  Then: W1 (hidden1 * input rows), b1 (hidden1 values),
        W2 (hidden2 * hidden1 rows), b2 (hidden2 values),
        W3 (hidden2 values), b3 (1 value)

Usage:
  python tools/export_mlp.py --model mlp_model.pt --output mlp.weights
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch MLP to text weights")
    parser.add_argument("--model", default="mlp_model.pt",
                        help="Input PyTorch model (default: mlp_model.pt)")
    parser.add_argument("--output", default="mlp.weights",
                        help="Output text weights (default: mlp.weights)")
    args = parser.parse_args()

    import torch

    state = torch.load(args.model, map_location="cpu", weights_only=True)

    # Detect model format: new (with BatchNorm at index 0) or legacy
    has_batchnorm = "0.weight" in state and "0.running_mean" in state

    if has_batchnorm:
        # New model: BN(0) -> Linear(1) -> LeakyReLU(2) -> Dropout(3) ->
        #            Linear(4) -> LeakyReLU(5) -> Dropout(6) -> Linear(7)
        bn_weight = state["0.weight"]
        bn_bias = state["0.bias"]
        bn_mean = state["0.running_mean"]
        bn_var = state["0.running_var"]
        bn_eps = 1e-5

        raw_w1 = state["1.weight"]
        raw_b1 = state["1.bias"]

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        shift = bn_bias - bn_mean * scale
        w1 = raw_w1 * scale.unsqueeze(0)
        b1 = raw_b1 + raw_w1 @ shift

        w2 = state["4.weight"]
        b2 = state["4.bias"]
        w3 = state["7.weight"]
        b3 = state["7.bias"]
    else:
        # Legacy model: Linear(0) -> ReLU(1) -> Linear(2) -> ReLU(3) -> Linear(4)
        w1 = state["0.weight"]
        b1 = state["0.bias"]
        w2 = state["2.weight"]
        b2 = state["2.bias"]
        w3 = state["4.weight"]
        b3 = state["4.bias"]

    input_dim = w1.shape[1]
    hidden1 = w1.shape[0]
    hidden2 = w2.shape[0]

    with open(args.output, "w") as f:
        f.write(f"{input_dim} {hidden1} {hidden2}\n")
        for row in w1:
            f.write(" ".join(f"{v:.8f}" for v in row.tolist()) + "\n")
        f.write(" ".join(f"{v:.8f}" for v in b1.tolist()) + "\n")
        for row in w2:
            f.write(" ".join(f"{v:.8f}" for v in row.tolist()) + "\n")
        f.write(" ".join(f"{v:.8f}" for v in b2.tolist()) + "\n")
        for v in w3.squeeze(0).tolist():
            f.write(f"{v:.8f}\n")
        f.write(f"{b3.item():.8f}\n")

    print(f"Exported {args.model} -> {args.output}")
    print(f"  Architecture: {input_dim} -> {hidden1} -> {hidden2} -> 1")
    if has_batchnorm:
        print(f"  BatchNorm fused into first Linear layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
