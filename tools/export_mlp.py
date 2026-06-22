#!/usr/bin/env python3
"""Export a PyTorch MLP model (mlp_model.pt) to a text weights file (mlp.weights).

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

    w1 = state["0.weight"]  # [64, 74]
    b1 = state["0.bias"]    # [64]
    w2 = state["2.weight"]  # [32, 64]
    b2 = state["2.bias"]    # [32]
    w3 = state["4.weight"]  # [1, 32]
    b3 = state["4.bias"]    # [1]

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
