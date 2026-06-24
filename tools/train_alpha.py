#!/usr/bin/env python3
"""Train ResNet-SE model for the Alpha engine from Floodgate CSA kifu files.

Usage:
  python train_alpha.py --kifu kifu/floodgate --model alpha_model.pt
  python train_alpha.py --kifu kifu/floodgate --model alpha_model.pt --min-rate 3000 --epochs 20
  python train_alpha.py --cache kifu_cache.npz --model alpha_model.pt --resume
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import List

from csa_parser import parse_game, sample_positions
from tqdm import tqdm


BOARD_SQUARES = 81
HAND_TYPES = 7
POLICY_SIZE = 2187
MAX_ATTACK = 8
BASE_STATE_SIZE = 96
STATE_SIZE = 258
INPUT_CHANNELS = 45

HAND_MAX_COUNTS = [18, 4, 4, 4, 4, 2, 2]


def import_torch():
    import torch
    from torch import nn
    return torch, nn


def pick_device(torch_mod, requested: str):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


class SEBlock:
    """Squeeze-and-Excitation block factory (created inside build_model)."""
    pass


def build_model(nn, channels=192, num_blocks=15, se_ratio=4):
    """Build ShogiResNetSE model."""
    import torch
    import torch.nn.functional as F

    class _SEBlock(nn.Module):
        def __init__(self, ch, ratio):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(ch, ch // ratio)
            self.fc2 = nn.Linear(ch // ratio, ch)

        def forward(self, x):
            b, c, _, _ = x.shape
            s = self.pool(x).view(b, c)
            s = F.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
            return x * s

    class _ResBlock(nn.Module):
        def __init__(self, ch, ratio):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)
            self.se = _SEBlock(ch, ratio)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            return F.relu(out + residual)

    class ShogiResNetSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels = channels
            self.input_conv = nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False)
            self.input_bn = nn.BatchNorm2d(channels)

            self.res_blocks = nn.Sequential(
                *[_ResBlock(channels, se_ratio) for _ in range(num_blocks)]
            )

            # Value head (WDL: win/draw/loss)
            self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(BOARD_SQUARES, 256)
            self.value_fc2 = nn.Linear(256, 3)

            # Policy head
            self.policy_conv = nn.Conv2d(channels, 27, 1)

        def forward(self, features):
            x = F.relu(self.input_bn(self.input_conv(features)))
            x = self.res_blocks(x)

            # Value
            v = F.relu(self.value_bn(self.value_conv(x)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            value_wdl = self.value_fc2(v)

            # Policy
            p = self.policy_conv(x)
            policy_logits = p.view(p.size(0), -1)

            return value_wdl, policy_logits

    return ShogiResNetSE()


def encoded_to_spatial(encoded_np: np.ndarray) -> np.ndarray:
    """Convert [N, 258] int encoded data to [N, 45, 9, 9] float spatial features.

    Channel layout:
      0-27: piece planes (14 piece types x 2 colors, one-hot)
      28-41: hand piece planes (7 types x 2 colors, normalized count)
      42-43: attack count planes (own/opponent, /8.0)
      44: side to move plane (1.0=Black, 0.0=White)
    """
    n = encoded_np.shape[0]
    features = np.zeros((n, INPUT_CHANNELS, 9, 9), dtype=np.float32)

    squares = encoded_np[:, :BOARD_SQUARES]

    for piece_id in range(1, 15):
        features[:, piece_id - 1, :, :] = (squares == piece_id).reshape(n, 9, 9)

    for piece_id in range(15, 29):
        features[:, 14 + (piece_id - 15), :, :] = (squares == piece_id).reshape(n, 9, 9)

    hand_start = BOARD_SQUARES
    for i in range(HAND_TYPES):
        max_c = HAND_MAX_COUNTS[i]
        bh = encoded_np[:, hand_start + i].astype(np.float32) / max_c
        features[:, 28 + i, :, :] = bh[:, None, None]
    for i in range(HAND_TYPES):
        max_c = HAND_MAX_COUNTS[i]
        wh = encoded_np[:, hand_start + HAND_TYPES + i].astype(np.float32) / max_c
        features[:, 35 + i, :, :] = wh[:, None, None]

    own_start = BASE_STATE_SIZE
    opp_start = own_start + BOARD_SQUARES
    if encoded_np.shape[1] >= opp_start + BOARD_SQUARES:
        own_atk = encoded_np[:, own_start:own_start + BOARD_SQUARES].astype(np.float32) / 8.0
        opp_atk = encoded_np[:, opp_start:opp_start + BOARD_SQUARES].astype(np.float32) / 8.0
        features[:, 42, :, :] = own_atk.reshape(n, 9, 9)
        features[:, 43, :, :] = opp_atk.reshape(n, 9, 9)

    side_col = BOARD_SQUARES + 2 * HAND_TYPES
    side = (encoded_np[:, side_col] == 0).astype(np.float32)
    features[:, 44, :, :] = side[:, None, None]

    return features


def load_games(kifu_dir: Path, min_rate: int, max_games: int,
               sample_rate: float, opening_n: int, endgame_n: int,
               parse_workers: int = 0) -> List[dict]:
    """Load and parse CSA kifu files into training samples."""
    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)
    if max_games > 0:
        csa_files = csa_files[:max_games]

    all_samples = []
    games_ok = 0
    skipped = 0

    worker_count = parse_workers if parse_workers > 0 else max(1, os.cpu_count() or 1)
    if worker_count <= 1:
        for filepath in tqdm(csa_files, desc="Parsing kifu", unit="file",
                             file=sys.stderr, miniters=1, dynamic_ncols=True):
            samples = parse_game(filepath, min_rate=min_rate)
            if samples is None:
                skipped += 1
                continue
            sampled = sample_positions(samples, sample_rate, opening_n, endgame_n)
            all_samples.extend(sampled)
            games_ok += 1
    else:
        print(f"Parsing workers: {worker_count}", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            sampled_iter = executor.map(
                _parse_and_sample_file,
                [str(p) for p in csa_files],
                repeat(min_rate),
                repeat(sample_rate),
                repeat(opening_n),
                repeat(endgame_n),
                chunksize=8,
            )
            for sampled in tqdm(sampled_iter, total=len(csa_files),
                                desc="Parsing kifu", unit="file",
                                file=sys.stderr, miniters=1, dynamic_ncols=True):
                if sampled is None:
                    skipped += 1
                    continue
                all_samples.extend(sampled)
                games_ok += 1

    print(f"Loaded {games_ok} games, {len(all_samples)} samples "
          f"(skipped {skipped})", file=sys.stderr)
    return all_samples


def _parse_and_sample_file(filepath: str, min_rate: int,
                           sample_rate: float, opening_n: int,
                           endgame_n: int):
    samples = parse_game(Path(filepath), min_rate=min_rate)
    if samples is None:
        return None
    return sample_positions(samples, sample_rate, opening_n, endgame_n)


def samples_to_tensors(samples: List[dict], torch_mod):
    """Convert parsed samples to spatial feature tensors (CPU)."""
    n = len(samples)
    print(f"  Extracting {n:,} samples...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    encoded_np = np.array([s["encoded"] for s in samples], dtype=np.int16)
    values = np.array([s["value"] for s in samples], dtype=np.float32)
    policies = np.array([s["move_index"] for s in samples], dtype=np.int64)
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    print("  Converting to spatial features...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_np = encoded_to_spatial(encoded_np)
    del encoded_np
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    print("  Converting to tensors...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_t = torch_mod.from_numpy(features_np)
    value_t = torch_mod.from_numpy(values)
    policy_t = torch_mod.from_numpy(policies)
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    return features_t, value_t, policy_t


def save_cache(path, samples):
    """Save parsed samples to numpy cache file."""
    actual_path = path if path.endswith(".npz") else path + ".npz"
    n = len(samples)
    print(f"Saving cache ({n:,} positions)...", file=sys.stderr)
    t0 = time.time()
    encoded = np.array([s["encoded"] for s in samples], dtype=np.uint8)
    values = np.array([s["value"] for s in samples], dtype=np.float32)
    policies = np.array([s["move_index"] for s in samples], dtype=np.int32)
    np.savez(path, encoded=encoded, values=values, policies=policies)
    size_mb = os.path.getsize(actual_path) / (1024 * 1024)
    print(f"  Saved: {actual_path} ({size_mb:.0f} MB, {time.time() - t0:.1f}s)",
          file=sys.stderr)


def load_cache(path, torch_mod):
    """Load cached data and convert to spatial feature tensors."""
    actual_path = path if path.endswith(".npz") else path + ".npz"
    if not os.path.exists(actual_path):
        return None

    print(f"Loading cache: {actual_path}...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    data = np.load(actual_path)
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    encoded = data["encoded"]
    n = len(encoded)
    print(f"  {n:,} positions", file=sys.stderr)

    print("  Converting to spatial features...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_np = encoded_to_spatial(encoded.astype(np.int16))
    del encoded
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    print("  Converting to tensors...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_t = torch_mod.from_numpy(features_np)
    value_t = torch_mod.from_numpy(data["values"].copy())
    policy_t = torch_mod.from_numpy(data["policies"].astype(np.int64))
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    return features_t, value_t, policy_t


def train_model(model, features_t, value_t, policy_t,
                torch_mod, nn, device, epochs: int, batch_size: int, lr: float,
                resumed: bool = False, loser_policy_weight: float = 0.1):
    """Train the ResNet-SE model with WDL value and policy targets."""
    model.train()
    max_lr = lr * 0.2 if resumed else lr
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)

    n = features_t.shape[0]
    total_batches = (n + batch_size - 1) // batch_size
    total_steps = total_batches * epochs

    scheduler = torch_mod.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100,
    )
    value_loss_fn = nn.CrossEntropyLoss()
    policy_loss_fn = nn.CrossEntropyLoss(reduction='none')

    # WDL targets from scalar value labels
    # value > 0 → win [1,0,0], value < 0 → loss [0,0,1], value == 0 → draw [0,1,0]
    wdl_targets = torch_mod.zeros(n, dtype=torch_mod.long)
    wdl_targets[value_t > 0] = 0   # win
    wdl_targets[value_t == 0] = 1  # draw
    wdl_targets[value_t < 0] = 2   # loss

    policy_weight = torch_mod.where(
        value_t > 0,
        torch_mod.ones_like(value_t),
        torch_mod.full_like(value_t, loser_policy_weight),
    )

    n_win = int((wdl_targets == 0).sum().item())
    n_draw = int((wdl_targets == 1).sum().item())
    n_loss = int((wdl_targets == 2).sum().item())
    mode = "fine-tune" if resumed else "fresh"
    print(f"Training ({mode}): {n:,} samples (W:{n_win:,}/D:{n_draw:,}/L:{n_loss:,}), "
          f"{epochs} epochs, batch={batch_size}, lr={max_lr:.2e}",
          file=sys.stderr)

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch_mod.randperm(n)
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b_feat = features_t[idx].to(device, non_blocking=True)
            b_wdl = wdl_targets[idx].to(device, non_blocking=True)
            b_pol = policy_t[idx].to(device, non_blocking=True)
            b_pw = policy_weight[idx].to(device, non_blocking=True)

            pred_wdl, pred_p = model(b_feat)
            loss_v = value_loss_fn(pred_wdl, b_wdl)
            per_sample_p = policy_loss_fn(pred_p, b_pol)
            loss_p = (per_sample_p * b_pw).mean()
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
            print(f"\r  epoch {epoch + 1}/{epochs}: [{bar}] {pct:3d}% | "
                  f"v={total_loss_v / batches:.4f} p={total_loss_p / batches:.4f}",
                  end="", file=sys.stderr, flush=True)

        elapsed = time.time() - t0
        avg_v = total_loss_v / batches
        avg_p = total_loss_p / batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\r  epoch {epoch + 1}/{epochs}: "
              f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f} "
              f"lr={current_lr:.6f} ({elapsed:.1f}s)          ", file=sys.stderr)

    return avg_v, avg_p


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-SE Alpha model from CSA kifu")
    parser.add_argument("--kifu", default="", help="Root directory of CSA kifu files")
    parser.add_argument("--model", default="alpha_model.pt", help="Model save path")
    parser.add_argument("--output", default="alpha_model.onnx", help="ONNX output path")
    parser.add_argument("--cache", default="", help="Cache file (.npz) for parsed kifu data")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-rate", type=int, default=2500)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument("--opening-n", type=int, default=40)
    parser.add_argument("--endgame-n", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parse-workers", type=int, default=0)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=15)
    args = parser.parse_args()

    torch_mod, nn = import_torch()
    device = pick_device(torch_mod, args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Load data
    cached = False
    if args.cache:
        result = load_cache(args.cache, torch_mod)
        if result is not None:
            features_t, value_t, policy_t = result
            cached = True

    if not cached:
        if not args.kifu:
            print("ERROR: --kifu required (no cache available)", file=sys.stderr)
            return 1
        samples = load_games(
            Path(args.kifu), args.min_rate, args.max_games,
            args.sample_rate, args.opening_n, args.endgame_n,
            args.parse_workers)
        if not samples:
            print("No samples loaded.", file=sys.stderr)
            return 1
        if args.cache:
            save_cache(args.cache, samples)
        features_t, value_t, policy_t = samples_to_tensors(samples, torch_mod)
        del samples

    # Build/load model
    model_path = Path(args.model)
    model = build_model(nn, channels=args.channels, num_blocks=args.blocks).to(device)
    resumed = False
    if args.resume and model_path.exists():
        state = torch_mod.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        resumed = True
        print(f"Resumed from {model_path}", file=sys.stderr)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.channels}ch x {args.blocks}blocks, {params:,} params", file=sys.stderr)

    # Multi-GPU
    gpu_count = torch_mod.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = torch_mod.nn.DataParallel(model)
        print(f"Using {gpu_count} GPUs via DataParallel", file=sys.stderr)

    # Train
    final_vloss, final_ploss = train_model(
        model, features_t, value_t, policy_t,
        torch_mod, nn, device, args.epochs, args.batch_size, args.lr,
        resumed=resumed)

    print(f"value_loss={final_vloss:.4f} policy_loss={final_ploss:.4f}")

    # Save
    save_model = model.module if hasattr(model, 'module') else model
    if model_path.parent != Path(""):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(save_model.state_dict(), model_path)
    print(f"Saved model to {model_path}", file=sys.stderr)

    # Export ONNX
    onnx_path = args.output
    print(f"Exporting ONNX to {onnx_path}...", file=sys.stderr)
    save_model = save_model.cpu()
    save_model.eval()
    dummy = torch_mod.zeros(1, INPUT_CHANNELS, 9, 9)
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
        torch_mod.onnx.export(
            save_model, dummy, onnx_path,
            dynamo=False, opset_version=17,
            input_names=["features"],
            output_names=["value_wdl", "policy_logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "value_wdl": {0: "batch"},
                "policy_logits": {0: "batch"},
            },
        )
        print(f"ONNX exported: {onnx_path}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
