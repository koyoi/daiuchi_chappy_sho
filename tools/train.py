#!/usr/bin/env python3
"""Train the Transformer model directly from Floodgate CSA kifu files.

Usage:
  python train.py --kifu kifu/floodgate --model nn_model.pt
  python train.py --kifu kifu/floodgate --model nn_model.pt --min-rate 2500 --epochs 20
  python train.py --kifu kifu/floodgate --model nn_model.pt --max-games 1000  # quick test
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import List

from csa_parser import parse_game, sample_positions


BOARD_SQUARES = 81
HAND_TYPES = 7
VOCAB_SIZE = 29
HAND_MAX = 19
SEQ_LEN = 95
POLICY_SIZE = 2187
STATE_SIZE = 96


def import_torch():
    import torch
    from torch import nn
    return torch, nn


def pick_device(torch_mod, requested: str):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


def build_model(nn, d_model=128, nhead=8, num_layers=4, dim_ff=256):
    """Build ShogiTransformer (same as nn_eval.py)."""
    import torch

    class ShogiTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.piece_embed = nn.Embedding(VOCAB_SIZE, d_model)
            self.pos_embed = nn.Embedding(SEQ_LEN, d_model)
            self.hand_count_embed = nn.Embedding(HAND_MAX + 1, d_model)
            self.side_embed = nn.Embedding(2, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                batch_first=True, dropout=0.1,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.value_head = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
            )
            self.policy_head = nn.Sequential(
                nn.Linear(d_model * BOARD_SQUARES, 512), nn.ReLU(), nn.Linear(512, POLICY_SIZE),
            )

        def forward(self, board_tokens, hand_tokens, side_token):
            B = board_tokens.shape[0]
            pos_ids = torch.arange(SEQ_LEN, device=board_tokens.device).unsqueeze(0).expand(B, -1)
            board_emb = self.piece_embed(board_tokens)
            hand_emb = self.hand_count_embed(hand_tokens)
            seq = torch.cat([board_emb, hand_emb], dim=1)
            seq = seq + self.pos_embed(pos_ids)
            side_emb = self.side_embed(side_token).unsqueeze(1)
            seq = seq + side_emb
            out = self.transformer(seq)
            global_repr = out.mean(dim=1)
            value = self.value_head(global_repr).squeeze(-1)
            board_out = out[:, :BOARD_SQUARES, :].reshape(B, -1)
            policy_logits = self.policy_head(board_out)
            return value, policy_logits

    return ShogiTransformerModel()


def load_games(kifu_dir: Path, min_rate: int, max_games: int,
               sample_rate: float, opening_n: int, endgame_n: int) -> List[dict]:
    """Load and parse CSA kifu files directly into training samples."""
    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)
    if max_games > 0:
        csa_files = csa_files[:max_games]

    all_samples = []
    games_ok = 0
    skipped = 0

    for i, filepath in enumerate(csa_files):
        if (i + 1) % 1000 == 0:
            print(f"  Parsing {i + 1}/{len(csa_files)}... ({len(all_samples)} samples)",
                  file=sys.stderr)

        samples = parse_game(filepath, min_rate=min_rate)
        if samples is None:
            skipped += 1
            continue

        sampled = sample_positions(samples, sample_rate, opening_n, endgame_n)
        all_samples.extend(sampled)
        games_ok += 1

    print(f"Loaded {games_ok} games, {len(all_samples)} samples "
          f"(skipped {skipped})", file=sys.stderr)
    return all_samples


def samples_to_tensors(samples: List[dict], torch_mod, device):
    """Convert parsed samples to training tensors."""
    n = len(samples)
    board_t = torch_mod.zeros(n, BOARD_SQUARES, dtype=torch_mod.long)
    hand_t = torch_mod.zeros(n, 2 * HAND_TYPES, dtype=torch_mod.long)
    side_t = torch_mod.zeros(n, dtype=torch_mod.long)
    value_t = torch_mod.zeros(n, dtype=torch_mod.float32)
    policy_t = torch_mod.zeros(n, dtype=torch_mod.long)

    for i, s in enumerate(samples):
        enc = s["encoded"]
        for j in range(BOARD_SQUARES):
            board_t[i, j] = max(0, min(enc[j], VOCAB_SIZE - 1))
        for j in range(2 * HAND_TYPES):
            hand_t[i, j] = max(0, min(enc[BOARD_SQUARES + j], HAND_MAX))
        side_t[i] = enc[-1]
        value_t[i] = s["value"]
        policy_t[i] = s["move_index"]

    return (board_t.to(device), hand_t.to(device), side_t.to(device),
            value_t.to(device), policy_t.to(device))


def train_model(model, board_t, hand_t, side_t, value_t, policy_t,
                torch_mod, nn, device, epochs: int, batch_size: int, lr: float):
    """Train the model on the given data."""
    model.train()
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    n = board_t.shape[0]
    print(f"Training: {n} samples, {epochs} epochs, batch={batch_size}, lr={lr}",
          file=sys.stderr)

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch_mod.randperm(n, device=device)
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        total_batches = (n + batch_size - 1) // batch_size
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred_v, pred_p = model(board_t[idx], hand_t[idx], side_t[idx])
            loss_v = value_loss_fn(pred_v, value_t[idx])
            loss_p = policy_loss_fn(pred_p, policy_t[idx])
            loss = loss_v + loss_p

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch_mod.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
            batches += 1

            pct = batches * 100 // total_batches
            filled = pct // 5
            bar = "=" * filled + ">" * (1 if filled < 20 else 0) + " " * (20 - filled - (1 if filled < 20 else 0))
            avg_v_so_far = total_loss_v / batches
            avg_p_so_far = total_loss_p / batches
            print(f"\r  epoch {epoch + 1}/{epochs}: [{bar}] {pct:3d}% | "
                  f"v={avg_v_so_far:.4f} p={avg_p_so_far:.4f}",
                  end="", file=sys.stderr, flush=True)

        scheduler.step()
        elapsed = time.time() - t0
        avg_v = total_loss_v / batches
        avg_p = total_loss_p / batches
        current_lr = scheduler.get_last_lr()[0]
        print(f"\r  epoch {epoch + 1}/{epochs}: "
              f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f} "
              f"lr={current_lr:.6f} ({elapsed:.1f}s)          ", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer from CSA kifu")
    parser.add_argument("--kifu", required=True, help="Root directory of CSA kifu files")
    parser.add_argument("--model", default="nn_model.pt", help="Model save path")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-rate", type=int, default=0,
                        help="Minimum player rating (0=all)")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games to load (0=all)")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="Middle-game position sampling rate")
    parser.add_argument("--opening-n", type=int, default=20,
                        help="Always include first N moves per game")
    parser.add_argument("--endgame-n", type=int, default=20,
                        help="Always include last N moves per game")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing model weights")
    args = parser.parse_args()

    torch_mod, nn = import_torch()
    device = pick_device(torch_mod, args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Load CSA kifu directly
    samples = load_games(
        Path(args.kifu), args.min_rate, args.max_games,
        args.sample_rate, args.opening_n, args.endgame_n,
    )
    if not samples:
        print("No samples loaded. Check kifu path and filters.", file=sys.stderr)
        return 1

    # Shuffle
    random.shuffle(samples)

    # Convert to tensors
    print("Converting to tensors...", file=sys.stderr)
    board_t, hand_t, side_t, value_t, policy_t = samples_to_tensors(
        samples, torch_mod, device)
    del samples  # free memory

    # Build/load model
    model_path = Path(args.model)
    model = build_model(nn).to(device)
    if args.resume and model_path.exists():
        state = torch_mod.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {model_path}", file=sys.stderr)

    # Train
    train_model(model, board_t, hand_t, side_t, value_t, policy_t,
                torch_mod, nn, device, args.epochs, args.batch_size, args.lr)

    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True) if model_path.parent != Path("") else None
    torch_mod.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}", file=sys.stderr)

    # Print model stats
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
