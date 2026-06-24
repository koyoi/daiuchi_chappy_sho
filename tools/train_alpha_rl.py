#!/usr/bin/env python3
"""Train ResNet-SE model from self-play data (reinforcement learning).

Uses MCTS visit distributions as policy targets and game outcomes as WDL targets.

Usage:
  python train_alpha_rl.py --data selfplay_data.npz --model alpha_model.pt
  python train_alpha_rl.py --data sp1.npz sp2.npz sp3.npz --model alpha_model.pt --buffer-size 2000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


POLICY_SIZE = 2187


def import_torch():
    import torch
    from torch import nn
    return torch, nn


def main():
    parser = argparse.ArgumentParser(description="RL training from self-play data")
    parser.add_argument("--data", nargs="+", required=True, help=".npz self-play data files")
    parser.add_argument("--model", default="alpha_model.pt", help="Model path (load & save)")
    parser.add_argument("--output", default="alpha_model.onnx", help="ONNX output path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=2000000,
                        help="Max positions in replay buffer (FIFO)")
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=15)
    args = parser.parse_args()

    torch_mod, nn = import_torch()
    from train_alpha import build_model, pick_device, encoded_to_spatial, INPUT_CHANNELS

    device = pick_device(torch_mod, args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Load all data files into replay buffer
    all_encoded = []
    all_policy = []
    all_wdl = []

    for data_path in args.data:
        print(f"Loading {data_path}...", end="", file=sys.stderr, flush=True)
        data = np.load(data_path)
        all_encoded.append(data["encoded"])
        all_policy.append(data["policy_target"])
        all_wdl.append(data["wdl_target"])
        print(f" {len(data['encoded'])} positions", file=sys.stderr)

    encoded = np.concatenate(all_encoded, axis=0)
    policy_target = np.concatenate(all_policy, axis=0)
    wdl_target = np.concatenate(all_wdl, axis=0)
    del all_encoded, all_policy, all_wdl

    # Apply buffer size limit (keep most recent)
    if len(encoded) > args.buffer_size:
        print(f"Trimming replay buffer: {len(encoded)} -> {args.buffer_size}", file=sys.stderr)
        encoded = encoded[-args.buffer_size:]
        policy_target = policy_target[-args.buffer_size:]
        wdl_target = wdl_target[-args.buffer_size:]

    n = len(encoded)
    print(f"Replay buffer: {n:,} positions", file=sys.stderr)

    # Convert to spatial features
    print("Converting to spatial features...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_np = encoded_to_spatial(encoded.astype(np.int16))
    del encoded
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    features_t = torch_mod.from_numpy(features_np)
    policy_t = torch_mod.from_numpy(policy_target)
    wdl_t = torch_mod.from_numpy(wdl_target)
    del features_np, policy_target, wdl_target

    # Build/load model
    model = build_model(nn, channels=args.channels, num_blocks=args.blocks).to(device)
    model_path = Path(args.model)
    if model_path.exists():
        state = torch_mod.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {model_path}", file=sys.stderr)
    else:
        print("WARNING: No model found, training from scratch", file=sys.stderr)

    # Training
    import torch.nn.functional as F

    model.train()
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_batches = (n + args.batch_size - 1) // args.batch_size
    total_steps = total_batches * args.epochs

    scheduler = torch_mod.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100,
    )

    wdl_loss_fn = nn.CrossEntropyLoss()

    n_win = int((wdl_t[:, 0] > 0.5).sum().item())
    n_draw = int((wdl_t[:, 1] > 0.5).sum().item())
    n_loss = int((wdl_t[:, 2] > 0.5).sum().item())
    print(f"Training (RL): {n:,} positions (W:{n_win:,}/D:{n_draw:,}/L:{n_loss:,}), "
          f"{args.epochs} epochs, batch={args.batch_size}, lr={args.lr:.2e}",
          file=sys.stderr)

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch_mod.randperm(n)
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            b_feat = features_t[idx].to(device, non_blocking=True)
            b_wdl = wdl_t[idx].to(device, non_blocking=True)
            b_policy = policy_t[idx].to(device, non_blocking=True)

            pred_wdl, pred_policy = model(b_feat)

            # WDL loss
            wdl_idx = b_wdl.argmax(dim=1)
            loss_v = wdl_loss_fn(pred_wdl, wdl_idx)

            # Policy loss: KL divergence with MCTS visit distribution
            log_pred = F.log_softmax(pred_policy, dim=1)
            loss_p = -(b_policy * log_pred).sum(dim=1).mean()

            loss = loss_v + loss_p

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch_mod.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
            batches += 1

            pct = batches * 100 // total_batches
            filled = pct // 5
            bar = "=" * filled + ">" * (1 if filled < 20 else 0) + " " * (20 - filled - (1 if filled < 20 else 0))
            print(f"\r  epoch {epoch + 1}/{args.epochs}: [{bar}] {pct:3d}% | "
                  f"v={total_loss_v / batches:.4f} p={total_loss_p / batches:.4f}",
                  end="", file=sys.stderr, flush=True)

        elapsed = time.time() - t0
        avg_v = total_loss_v / batches
        avg_p = total_loss_p / batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\r  epoch {epoch + 1}/{args.epochs}: "
              f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f} "
              f"lr={current_lr:.6f} ({elapsed:.1f}s)          ", file=sys.stderr)

    print(f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f}")

    # Save
    if model_path.parent != Path(""):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}", file=sys.stderr)

    # Export ONNX
    print(f"Exporting ONNX to {args.output}...", file=sys.stderr)
    model = model.cpu()
    model.eval()
    dummy = torch_mod.zeros(1, INPUT_CHANNELS, 9, 9)
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
        torch_mod.onnx.export(
            model, dummy, args.output,
            dynamo=False, opset_version=17,
            input_names=["features"],
            output_names=["value_wdl", "policy_logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "value_wdl": {0: "batch"},
                "policy_logits": {0: "batch"},
            },
        )
        print(f"ONNX exported: {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
