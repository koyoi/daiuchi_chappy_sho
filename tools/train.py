#!/usr/bin/env python3
"""Train the Transformer model directly from Floodgate CSA kifu files.

Usage:
  python train.py --kifu kifu/floodgate --model nn_model.pt
  python train.py --kifu kifu/floodgate --model nn_model.pt --min-rate 2500 --epochs 20
  python train.py --kifu kifu/floodgate --model nn_model.pt --max-games 1000  # quick test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import List

from csa_parser import parse_game, sample_positions
from tqdm import tqdm


BOARD_SQUARES = 81
HAND_TYPES = 7
VOCAB_SIZE = 29
HAND_MAX = 19
SEQ_LEN = 95
POLICY_SIZE = 2187
MAX_ATTACK = 8
BASE_STATE_SIZE = 96
STATE_SIZE = 258


def import_torch():
    import torch
    from torch import nn
    return torch, nn


def pick_device(torch_mod, requested: str):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


def build_model(nn, d_model=128, nhead=8, num_layers=4, dim_ff=512):
    """Build ShogiTransformer (same as nn_eval.py)."""
    import torch
    import torch.nn.functional as F

    class ShogiTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.piece_embed = nn.Embedding(VOCAB_SIZE, d_model)
            self.file_embed = nn.Embedding(9, d_model)
            self.rank_embed = nn.Embedding(9, d_model)
            self.hand_pos_embed = nn.Embedding(2 * HAND_TYPES, d_model)
            self.hand_count_embed = nn.Embedding(HAND_MAX + 1, d_model)
            self.side_embed = nn.Embedding(2, d_model)
            self.own_attack_embed = nn.Embedding(MAX_ATTACK + 1, d_model)
            self.opp_attack_embed = nn.Embedding(MAX_ATTACK + 1, d_model)
            self.cnn = nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, 3, padding=1),
            )
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

        def forward(self, board_tokens, hand_tokens, side_token,
                    own_atk=None, opp_atk=None):
            B = board_tokens.shape[0]
            dev = board_tokens.device
            d = self.d_model

            board_emb = self.piece_embed(board_tokens)
            files = torch.arange(BOARD_SQUARES, device=dev) % 9
            ranks = torch.arange(BOARD_SQUARES, device=dev) // 9
            board_emb = board_emb + self.file_embed(files) + self.rank_embed(ranks)

            if own_atk is not None:
                board_emb = board_emb + self.own_attack_embed(own_atk)
            if opp_atk is not None:
                board_emb = board_emb + self.opp_attack_embed(opp_atk)

            residual = board_emb
            x = board_emb.transpose(1, 2).reshape(B, d, 9, 9)
            x = self.cnn(x)
            x = x.reshape(B, d, BOARD_SQUARES).transpose(1, 2)
            board_emb = F.relu(x + residual)

            hand_emb = self.hand_count_embed(hand_tokens)
            hand_ids = torch.arange(2 * HAND_TYPES, device=dev)
            hand_emb = hand_emb + self.hand_pos_embed(hand_ids)

            seq = torch.cat([board_emb, hand_emb], dim=1)
            seq = seq + self.side_embed(side_token).unsqueeze(1)

            out = self.transformer(seq)
            global_repr = out.mean(dim=1)
            value = self.value_head(global_repr).squeeze(-1)
            board_out = out[:, :BOARD_SQUARES, :].reshape(B, -1)
            policy_logits = self.policy_head(board_out)
            return value, policy_logits

    return ShogiTransformerModel()


def load_games(kifu_dir: Path, min_rate: int, max_games: int,
               sample_rate: float, opening_n: int, endgame_n: int,
               parse_workers: int = 0) -> List[dict]:
    """Load and parse CSA kifu files directly into training samples."""
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
    """Worker entrypoint for process-parallel CSA parsing."""
    samples = parse_game(Path(filepath), min_rate=min_rate)
    if samples is None:
        return None
    return sample_positions(samples, sample_rate, opening_n, endgame_n)


def samples_to_tensors(samples: List[dict], torch_mod, device,
                       tensor_threads: int = 0):
    """Convert parsed samples to training tensors."""
    if tensor_threads > 0:
        torch_mod.set_num_threads(tensor_threads)

    n = len(samples)
    encoded_rows = [s["encoded"] for s in tqdm(samples, desc="Converting to tensors",
                                                unit="pos", file=sys.stderr, miniters=1,
                                                dynamic_ncols=True)]
    encoded_t = torch_mod.tensor(encoded_rows, dtype=torch_mod.long)

    board_t = encoded_t[:, :BOARD_SQUARES].clamp_(0, VOCAB_SIZE - 1)
    hand_start = BOARD_SQUARES
    hand_end = BOARD_SQUARES + 2 * HAND_TYPES
    hand_t = encoded_t[:, hand_start:hand_end].clamp_(0, HAND_MAX)
    side_t = encoded_t[:, hand_end]

    own_start = BASE_STATE_SIZE
    own_end = BASE_STATE_SIZE + BOARD_SQUARES
    opp_start = own_end
    opp_end = own_end + BOARD_SQUARES

    if encoded_t.size(1) >= opp_end:
        own_atk_t = encoded_t[:, own_start:own_end].clamp_(0, MAX_ATTACK)
        opp_atk_t = encoded_t[:, opp_start:opp_end].clamp_(0, MAX_ATTACK)
    else:
        own_atk_t = torch_mod.zeros(n, BOARD_SQUARES, dtype=torch_mod.long)
        opp_atk_t = torch_mod.zeros(n, BOARD_SQUARES, dtype=torch_mod.long)

    value_t = torch_mod.tensor([s["value"] for s in samples], dtype=torch_mod.float32)
    policy_t = torch_mod.tensor([s["move_index"] for s in samples], dtype=torch_mod.long)

    return (board_t.to(device), hand_t.to(device), side_t.to(device),
            own_atk_t.to(device), opp_atk_t.to(device),
            value_t.to(device), policy_t.to(device))


def train_model(model, board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t,
                torch_mod, nn, device, epochs: int, batch_size: int, lr: float,
                resumed: bool = False, loser_policy_weight: float = 0.1,
                round_number: int = 1):
    """Train the model on the given data."""
    model.train()
    round_decay = 0.95 ** (round_number - 1) if resumed else 1.0
    max_lr = lr * max(round_decay, 0.1)
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)

    n = board_t.shape[0]
    total_batches = (n + batch_size - 1) // batch_size
    total_steps = total_batches * epochs

    scheduler = torch_mod.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss(reduction='none')
    avg_v, avg_p = 1.0, 8.0

    # value_t: +1.0 = moving side won, -1.0 = moving side lost
    policy_weight_t = torch_mod.where(
        value_t > 0,
        torch_mod.ones_like(value_t),
        torch_mod.full_like(value_t, loser_policy_weight),
    )
    n_winner = int((value_t > 0).sum().item())
    n_loser = n - n_winner

    init_lr = max_lr / 10
    min_lr = init_lr / 100
    mode = "fine-tune" if resumed else "fresh"
    print(f"Training ({mode} R{round_number}): {n} samples ({n_winner} winner, {n_loser} loser), "
          f"{epochs} epochs, batch={batch_size}, "
          f"lr={init_lr:.2e}→{max_lr:.2e}→{min_lr:.2e}, "
          f"loser_pw={loser_policy_weight}",
          file=sys.stderr)

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch_mod.randperm(n, device=device)
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred_v, pred_p = model(board_t[idx], hand_t[idx], side_t[idx],
                                   own_atk_t[idx], opp_atk_t[idx])
            loss_v = value_loss_fn(pred_v, value_t[idx])
            per_sample_p = policy_loss_fn(pred_p, policy_t[idx])
            loss_p = (per_sample_p * policy_weight_t[idx]).mean()
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
            avg_v_so_far = total_loss_v / batches
            avg_p_so_far = total_loss_p / batches
            print(f"\r  epoch {epoch + 1}/{epochs}: [{bar}] {pct:3d}% | "
                  f"v={avg_v_so_far:.4f} p={avg_p_so_far:.4f}",
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
    parser = argparse.ArgumentParser(description="Train Transformer from CSA kifu")
    parser.add_argument("--kifu", required=True, help="Root directory of CSA kifu files")
    parser.add_argument("--model", default="nn_model.pt", help="Model save path")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-rate", type=int, default=2500,
                        help="Minimum player rating for both players (default: 2500)")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games to load (0=all)")
    parser.add_argument("--sample-rate", type=float, default=0.5,
                        help="Middle-game position sampling rate")
    parser.add_argument("--opening-n", type=int, default=40,
                        help="Always include first N moves per game")
    parser.add_argument("--endgame-n", type=int, default=40,
                        help="Always include last N moves per game")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing model weights")
    parser.add_argument("--round", type=int, default=1,
                        help="Current training round (for LR decay)")
    parser.add_argument("--parse-workers", type=int, default=0,
                        help="Parallel workers for kifu parsing (0=auto)")
    parser.add_argument("--tensor-threads", type=int, default=0,
                        help="CPU threads used during tensor conversion (0=default)")
    args = parser.parse_args()

    torch_mod, nn = import_torch()
    device = pick_device(torch_mod, args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Load CSA kifu directly
    samples = load_games(
        Path(args.kifu), args.min_rate, args.max_games,
        args.sample_rate, args.opening_n, args.endgame_n,
        args.parse_workers,
    )
    if not samples:
        print("No samples loaded. Check kifu path and filters.", file=sys.stderr)
        return 1

    # Shuffle
    random.shuffle(samples)

    # Convert to tensors
    board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t = samples_to_tensors(
        samples, torch_mod, device, args.tensor_threads)
    del samples  # free memory

    # Build/load model
    model_path = Path(args.model)
    model = build_model(nn).to(device)
    resumed = False
    if args.resume and model_path.exists():
        state = torch_mod.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        resumed = True
        print(f"Resumed from {model_path}", file=sys.stderr)

    # Multi-GPU
    gpu_count = torch_mod.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = torch_mod.nn.DataParallel(model)
        print(f"Using {gpu_count} GPUs via DataParallel", file=sys.stderr)

    effective_lr = args.lr * 0.2 if resumed else args.lr

    # Train
    final_vloss, final_ploss = train_model(
        model, board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t,
        torch_mod, nn, device, args.epochs, args.batch_size, effective_lr,
        resumed=resumed, round_number=args.round)

    # Summary to stdout (parsed by train_loop.py)
    print(f"value_loss={final_vloss:.4f} policy_loss={final_ploss:.4f}")

    # Save (unwrap DataParallel)
    save_model = model.module if hasattr(model, 'module') else model
    model_path.parent.mkdir(parents=True, exist_ok=True) if model_path.parent != Path("") else None
    torch_mod.save(save_model.state_dict(), model_path)
    print(f"Saved model to {model_path}", file=sys.stderr)

    # Print model stats
    params = sum(p.numel() for p in save_model.parameters())
    print(f"Model parameters: {params:,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
