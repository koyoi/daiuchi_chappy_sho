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
import numpy as np
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
    """Build ShogiTransformer with Relative Position Encoding."""
    import torch
    import torch.nn.functional as F

    d_head = d_model // nhead
    seq_len = BOARD_SQUARES + 2 * HAND_TYPES  # 95

    class RelPosEncoderLayer(nn.Module):
        """Post-norm Transformer encoder layer with relative position bias."""
        def __init__(self):
            super().__init__()
            self.nhead = nhead
            self.d_head = d_head
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_ff), nn.ReLU(),
                nn.Linear(dim_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn_drop = nn.Dropout(0.1)
            self.drop1 = nn.Dropout(0.1)
            self.drop2 = nn.Dropout(0.1)
            self.scale = d_head ** -0.5

        def forward(self, x, rel_bias):
            B, L, D = x.shape
            qkv = self.qkv_proj(x).reshape(B, L, 3, self.nhead, self.d_head)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale + rel_bias
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, L, D)
            out = self.out_proj(out)
            x = self.norm1(x + self.drop1(out))
            x = self.norm2(x + self.drop2(self.ff(x)))
            return x

    class ShogiTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.piece_embed = nn.Embedding(VOCAB_SIZE, d_model)
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
            self.layers = nn.ModuleList([RelPosEncoderLayer() for _ in range(num_layers)])
            self.rel_bias_table = nn.Parameter(torch.zeros(nhead, 17, 17))
            self.board_hand_bias = nn.Parameter(torch.zeros(nhead))
            self.hand_board_bias = nn.Parameter(torch.zeros(nhead))
            self.hand_hand_bias = nn.Parameter(torch.zeros(nhead))
            self.value_head = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
            )
            self.policy_head = nn.Sequential(
                nn.Linear(d_model * BOARD_SQUARES, 512), nn.ReLU(), nn.Linear(512, POLICY_SIZE),
            )
            board_pos = torch.arange(BOARD_SQUARES)
            rel_file = (board_pos % 9).unsqueeze(0) - (board_pos % 9).unsqueeze(1) + 8
            rel_rank = (board_pos // 9).unsqueeze(0) - (board_pos // 9).unsqueeze(1) + 8
            self.register_buffer('rel_file_idx', rel_file)
            self.register_buffer('rel_rank_idx', rel_rank)

        def _build_rel_bias(self):
            B2B = self.rel_bias_table[:, self.rel_rank_idx, self.rel_file_idx]
            bias = B2B.new_zeros(self.nhead, seq_len, seq_len)
            bias[:, :BOARD_SQUARES, :BOARD_SQUARES] = B2B
            bias[:, :BOARD_SQUARES, BOARD_SQUARES:] = self.board_hand_bias.unsqueeze(-1).unsqueeze(-1)
            bias[:, BOARD_SQUARES:, :BOARD_SQUARES] = self.hand_board_bias.unsqueeze(-1).unsqueeze(-1)
            bias[:, BOARD_SQUARES:, BOARD_SQUARES:] = self.hand_hand_bias.unsqueeze(-1).unsqueeze(-1)
            return bias

        def freeze_rel_bias(self):
            """Pre-compute rel_bias as a buffer for ONNX export (avoids advanced indexing)."""
            self.register_buffer('_frozen_rel_bias', self._build_rel_bias().detach())

        def forward(self, board_tokens, hand_tokens, side_token,
                    own_atk=None, opp_atk=None):
            B = board_tokens.shape[0]
            d = self.d_model

            board_emb = self.piece_embed(board_tokens)
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
            hand_ids = torch.arange(2 * HAND_TYPES, device=board_tokens.device)
            hand_emb = hand_emb + self.hand_pos_embed(hand_ids)

            seq = torch.cat([board_emb, hand_emb], dim=1)
            seq = seq + self.side_embed(side_token).unsqueeze(1)

            if hasattr(self, '_frozen_rel_bias'):
                rel_bias = self._frozen_rel_bias
            else:
                rel_bias = self._build_rel_bias()
            for layer in self.layers:
                seq = layer(seq, rel_bias)

            global_repr = seq.mean(dim=1)
            value = self.value_head(global_repr).squeeze(-1)
            board_out = seq[:, :BOARD_SQUARES, :].reshape(B, -1)
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


def samples_to_tensors(samples: List[dict], torch_mod, device=None,
                       tensor_threads: int = 0):
    """Convert parsed samples to CPU tensors (kept on CPU for per-batch GPU transfer)."""
    if tensor_threads > 0:
        torch_mod.set_num_threads(tensor_threads)

    n = len(samples)
    print(f"  Extracting fields from {n:,} samples...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    encoded_rows = [s["encoded"] for s in samples]
    values = [s["value"] for s in samples]
    policies = [s["move_index"] for s in samples]
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    cols = len(encoded_rows[0])
    print(f"  Building numpy array ({n:,} x {cols})...",
          end="", file=sys.stderr, flush=True)
    t0 = time.time()
    encoded_np = np.array(encoded_rows, dtype=np.int16)
    del encoded_rows
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    print("  Converting to tensors...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    board_t, hand_t, side_t, own_atk_t, opp_atk_t = _encoded_np_to_tensors(
        encoded_np, n, torch_mod)
    del encoded_np

    value_t = torch_mod.from_numpy(np.array(values, dtype=np.float32))
    policy_t = torch_mod.from_numpy(np.array(policies, dtype=np.int64))
    del values, policies
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    return (board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t)


def _encoded_np_to_tensors(encoded_np, n, torch_mod):
    """Slice encoded numpy array into individual CPU tensors."""
    hand_start = BOARD_SQUARES
    hand_end = BOARD_SQUARES + 2 * HAND_TYPES
    own_start = BASE_STATE_SIZE
    own_end = BASE_STATE_SIZE + BOARD_SQUARES
    opp_end = own_end + BOARD_SQUARES

    board_t = torch_mod.from_numpy(
        encoded_np[:, :BOARD_SQUARES].astype(np.int64)).clamp_(0, VOCAB_SIZE - 1)
    hand_t = torch_mod.from_numpy(
        encoded_np[:, hand_start:hand_end].astype(np.int64)).clamp_(0, HAND_MAX)
    side_t = torch_mod.from_numpy(
        encoded_np[:, hand_end].astype(np.int64))

    if encoded_np.shape[1] >= opp_end:
        own_atk_t = torch_mod.from_numpy(
            encoded_np[:, own_start:own_end].astype(np.int64)).clamp_(0, MAX_ATTACK)
        opp_atk_t = torch_mod.from_numpy(
            encoded_np[:, own_end:opp_end].astype(np.int64)).clamp_(0, MAX_ATTACK)
    else:
        own_atk_t = torch_mod.zeros(n, BOARD_SQUARES, dtype=torch_mod.long)
        opp_atk_t = torch_mod.zeros(n, BOARD_SQUARES, dtype=torch_mod.long)

    return board_t, hand_t, side_t, own_atk_t, opp_atk_t


def _parse_file_no_sample(filepath: str, min_rate: int):
    """Worker for parallel parsing without position sampling."""
    return parse_game(Path(filepath), min_rate=min_rate)


def load_games_grouped(kifu_dir: Path, min_rate: int, max_games: int,
                       parse_workers: int = 0):
    """Load ALL positions from games, preserving game boundaries for eval_drop.

    Returns (all_samples, game_slices) where game_slices is a list of
    (start_idx, end_idx) tuples with positions in ply order per game.
    """
    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)
    if max_games > 0:
        csa_files = csa_files[:max_games]

    all_samples: List[dict] = []
    game_slices: List[tuple] = []
    games_ok = 0
    skipped = 0

    worker_count = parse_workers if parse_workers > 0 else max(1, os.cpu_count() or 1)
    if worker_count <= 1:
        for filepath in tqdm(csa_files, desc="Parsing kifu (grouped)", unit="file",
                             file=sys.stderr, miniters=1, dynamic_ncols=True):
            samples = parse_game(filepath, min_rate=min_rate)
            if samples is None:
                skipped += 1
                continue
            start = len(all_samples)
            all_samples.extend(samples)
            game_slices.append((start, len(all_samples)))
            games_ok += 1
    else:
        print(f"Parsing workers: {worker_count}", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results_iter = executor.map(
                _parse_file_no_sample,
                [str(p) for p in csa_files],
                repeat(min_rate),
                chunksize=8,
            )
            for samples in tqdm(results_iter, total=len(csa_files),
                                desc="Parsing kifu (grouped)", unit="file",
                                file=sys.stderr, miniters=1, dynamic_ncols=True):
                if samples is None:
                    skipped += 1
                    continue
                start = len(all_samples)
                all_samples.extend(samples)
                game_slices.append((start, len(all_samples)))
                games_ok += 1

    print(f"Loaded {games_ok} games, {len(all_samples)} positions "
          f"(skipped {skipped})", file=sys.stderr)
    return all_samples, game_slices


def save_cache(path, samples, game_slices=None):
    """Save parsed samples to numpy cache file for fast reloading."""
    actual_path = path if path.endswith(".npz") else path + ".npz"
    n = len(samples)

    print(f"Saving cache ({n:,} positions)...", file=sys.stderr)
    t0 = time.time()
    encoded = np.array([s["encoded"] for s in samples], dtype=np.uint8)
    values = np.array([s["value"] for s in samples], dtype=np.float32)
    policies = np.array([s["move_index"] for s in samples], dtype=np.int32)

    arrays = dict(encoded=encoded, values=values, policies=policies)
    if game_slices is not None:
        arrays["game_slices"] = np.array(game_slices, dtype=np.int64)

    np.savez(path, **arrays)
    size_mb = os.path.getsize(actual_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Saved: {actual_path} ({size_mb:.0f} MB, {elapsed:.1f}s)", file=sys.stderr)


def load_cache(path, torch_mod):
    """Load cached numpy data and convert to CPU tensors.

    Returns ((board_t, hand_t, ..., value_t, policy_t), game_slices)
    or (None, None) if cache doesn't exist.
    """
    actual_path = path if path.endswith(".npz") else path + ".npz"
    if not os.path.exists(actual_path):
        return None, None

    print(f"Loading cache: {actual_path}...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    data = np.load(actual_path)
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    encoded = data["encoded"]
    n = len(encoded)
    print(f"  {n:,} positions, converting to tensors...", end="", file=sys.stderr, flush=True)
    t0 = time.time()

    board_t, hand_t, side_t, own_atk_t, opp_atk_t = _encoded_np_to_tensors(
        encoded, n, torch_mod)
    del encoded

    value_t = torch_mod.from_numpy(data["values"].copy())
    policy_t = torch_mod.from_numpy(data["policies"].astype(np.int64))
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    game_slices = None
    if "game_slices" in data:
        gs = data["game_slices"]
        game_slices = [(int(gs[i, 0]), int(gs[i, 1])) for i in range(len(gs))]

    return (board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t), game_slices


def evaluate_positions(model, board_t, hand_t, side_t, own_atk_t, opp_atk_t,
                       torch_mod, eval_batch=8192):
    """Batch inference on all positions (CPU tensors), returns value predictions on CPU."""
    model.eval()
    dev = next(model.parameters()).device
    n = board_t.shape[0]
    evals = torch_mod.zeros(n)
    total = (n + eval_batch - 1) // eval_batch
    with torch_mod.no_grad():
        for start in tqdm(range(0, n, eval_batch), desc="  Evaluating",
                          total=total, file=sys.stderr, dynamic_ncols=True):
            end = min(start + eval_batch, n)
            v, _ = model(
                board_t[start:end].to(dev), hand_t[start:end].to(dev),
                side_t[start:end].to(dev), own_atk_t[start:end].to(dev),
                opp_atk_t[start:end].to(dev))
            evals[start:end] = v.cpu()
    return evals


def compute_bootstrap_labels(evals, game_result_t, game_slices, torch_mod,
                              blend=0.5, drop_margin=0.8):
    """Compute improved value labels and policy weights from eval_drop.

    eval_drop[i] = evals[i] + evals[i+1] for consecutive positions in same game.
    Positive eval_drop = the mover made the position worse (blunder).

    Returns (new_value_t, policy_weight_t) as CPU tensors.
    """
    n = len(evals)

    eval_drop = torch_mod.zeros(n)
    if n > 1:
        eval_drop[:-1] = evals[:-1] + evals[1:]
    for _, end in game_slices:
        if end > 0:
            eval_drop[end - 1] = 0.0

    new_value = blend * game_result_t + (1.0 - blend) * evals
    new_value = new_value.clamp(-1.0, 1.0)

    policy_weight = torch_mod.clamp(
        1.0 - torch_mod.relu(eval_drop) / drop_margin, min=0.05)

    blunders = int((eval_drop > 0.5).sum().item())
    good_moves = int((eval_drop < -0.3).sum().item())
    print(f"  eval_drop: mean={eval_drop.mean():.4f} std={eval_drop.std():.4f} "
          f"blunders(>0.5)={blunders:,} good(<-0.3)={good_moves:,}",
          file=sys.stderr)
    print(f"  policy_weight: mean={policy_weight.mean():.3f} "
          f"[{policy_weight.min():.3f}, {policy_weight.max():.3f}]",
          file=sys.stderr)

    return new_value, policy_weight


def train_model(model, board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t,
                torch_mod, nn, device, epochs: int, batch_size: int, lr: float,
                resumed: bool = False, loser_policy_weight: float = 0.1,
                round_number: int = 1, policy_weight_t=None,
                patience: int = 0):
    """Train the model on the given data.

    patience: early stop after this many epochs without val loss improvement (0=disabled).
    """
    model.train()
    round_decay = 0.95 ** (round_number - 1) if resumed else 1.0
    max_lr = lr * max(round_decay, 0.1)

    n = board_t.shape[0]
    val_frac = 0.05 if n >= 100000 else 0.0
    n_val = int(n * val_frac)
    n_train = n - n_val

    if n_val > 0:
        all_perm = torch_mod.randperm(n)
        val_idx = all_perm[:n_val]
        train_idx = all_perm[n_val:]
    else:
        train_idx = torch_mod.arange(n)
        val_idx = None

    train_batches = (n_train + batch_size - 1) // batch_size
    total_steps = train_batches * epochs

    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
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

    if policy_weight_t is None:
        policy_weight_t = torch_mod.where(
            value_t > 0,
            torch_mod.ones_like(value_t),
            torch_mod.full_like(value_t, loser_policy_weight),
        )
        pw_desc = f"loser_pw={loser_policy_weight}"
    else:
        pw_desc = f"bootstrap pw_mean={policy_weight_t.mean():.3f}"
    n_positive = int((value_t > 0).sum().item())
    n_negative = n - n_positive

    init_lr = max_lr / 10
    min_lr = init_lr / 100
    mode = "fine-tune" if resumed else "fresh"
    val_str = f", val={n_val}" if n_val > 0 else ""
    patience_str = f", patience={patience}" if patience > 0 else ""
    print(f"Training ({mode} R{round_number}): {n_train} train{val_str} samples "
          f"({n_positive}+/{n_negative}-), "
          f"{epochs} epochs, batch={batch_size}, "
          f"lr={init_lr:.2e}→{max_lr:.2e}→{min_lr:.2e}, "
          f"{pw_desc}{patience_str}",
          file=sys.stderr)

    best_val_loss = float('inf')
    best_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        t0 = time.time()
        perm = train_idx[torch_mod.randperm(n_train)]
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        model.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            b_board = board_t[idx].to(device, non_blocking=True)
            b_hand = hand_t[idx].to(device, non_blocking=True)
            b_side = side_t[idx].to(device, non_blocking=True)
            b_own = own_atk_t[idx].to(device, non_blocking=True)
            b_opp = opp_atk_t[idx].to(device, non_blocking=True)
            b_val = value_t[idx].to(device, non_blocking=True)
            b_pol = policy_t[idx].to(device, non_blocking=True)
            b_pw = policy_weight_t[idx].to(device, non_blocking=True)

            pred_v, pred_p = model(b_board, b_hand, b_side, b_own, b_opp)
            loss_v = value_loss_fn(pred_v, b_val)
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

            pct = batches * 100 // train_batches
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

        val_str = ""
        if val_idx is not None:
            model.eval()
            val_v_total = 0.0
            val_p_total = 0.0
            val_batches = 0
            with torch_mod.no_grad():
                for vs in range(0, n_val, batch_size):
                    vi = val_idx[vs:vs + batch_size]
                    pv, pp = model(
                        board_t[vi].to(device), hand_t[vi].to(device),
                        side_t[vi].to(device), own_atk_t[vi].to(device),
                        opp_atk_t[vi].to(device))
                    val_v_total += value_loss_fn(pv, value_t[vi].to(device)).item()
                    val_p_total += policy_loss_fn(pp, policy_t[vi].to(device)).mean().item()
                    val_batches += 1
            val_v = val_v_total / val_batches
            val_p = val_p_total / val_batches
            val_loss = val_v + val_p
            val_str = f" | val: v={val_v:.4f} p={val_p:.4f}"

            if patience > 0:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    save_m = model.module if hasattr(model, 'module') else model
                    best_state = {k: v.clone() for k, v in save_m.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1

        print(f"\r  epoch {epoch + 1}/{epochs}: "
              f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f} "
              f"lr={current_lr:.6f}{val_str} ({elapsed:.1f}s)          ", file=sys.stderr)

        if patience > 0 and stale_epochs >= patience:
            print(f"  Early stop: no improvement for {patience} epochs", file=sys.stderr)
            break

    if best_state is not None:
        save_m = model.module if hasattr(model, 'module') else model
        save_m.load_state_dict(best_state)
        print(f"  Restored best val checkpoint (loss={best_val_loss:.4f})", file=sys.stderr)

    return avg_v, avg_p


def main():
    parser = argparse.ArgumentParser(description="Train Transformer from CSA kifu")
    parser.add_argument("--kifu", default="", help="Root directory of CSA kifu files")
    parser.add_argument("--model", default="nn_model.pt", help="Model save path")
    parser.add_argument("--output", default="nn_model.onnx",
                        help="ONNX output path (default: nn_model.onnx)")
    parser.add_argument("--cache", default="",
                        help="Cache file (.npz) for parsed kifu data")
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
    parser.add_argument("--bootstrap-rounds", type=int, default=0,
                        help="Self-evaluation bootstrap rounds (0=disabled)")
    parser.add_argument("--bootstrap-blend", type=float, default=0.5,
                        help="Blend: 0=model eval only, 1=game result only (default: 0.5)")
    parser.add_argument("--drop-margin", type=float, default=0.8,
                        help="eval_drop at which policy weight reaches min (default: 0.8)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stop after N epochs without val loss improvement (0=disabled)")
    args = parser.parse_args()

    torch_mod, nn = import_torch()
    device = pick_device(torch_mod, args.device)
    print(f"Device: {device}", file=sys.stderr)

    bootstrap = args.bootstrap_rounds > 0
    game_slices = None

    # Load data: from cache or parse kifu
    cached = False
    if args.cache:
        result, game_slices = load_cache(args.cache, torch_mod)
        if result is not None:
            board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t = result
            cached = True

    if not cached:
        if not args.kifu:
            print("ERROR: --kifu required (no cache available)", file=sys.stderr)
            return 1
        if bootstrap or args.cache:
            samples, game_slices = load_games_grouped(
                Path(args.kifu), args.min_rate, args.max_games, args.parse_workers)
        else:
            samples = load_games(
                Path(args.kifu), args.min_rate, args.max_games,
                args.sample_rate, args.opening_n, args.endgame_n,
                args.parse_workers)
            game_slices = None
        if not samples:
            print("No samples loaded. Check kifu path and filters.", file=sys.stderr)
            return 1
        if args.cache:
            save_cache(args.cache, samples, game_slices)
        board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t = samples_to_tensors(
            samples, torch_mod, tensor_threads=args.tensor_threads)
        del samples

    # Build/load model
    model_path = Path(args.model)
    model = build_model(nn).to(device)
    resumed = False
    if args.resume and model_path.exists():
        state = torch_mod.load(model_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(state)
            resumed = True
            print(f"Resumed from {model_path}", file=sys.stderr)
        except RuntimeError:
            print(f"Resume: architecture mismatch, training from scratch",
                  file=sys.stderr)
            model = build_model(nn).to(device)

    # Multi-GPU
    gpu_count = torch_mod.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = torch_mod.nn.DataParallel(model)
        print(f"Using {gpu_count} GPUs via DataParallel", file=sys.stderr)

    effective_lr = args.lr * 0.2 if resumed else args.lr

    if bootstrap:
        game_result_t = value_t.clone()
        total_rounds = 1 + args.bootstrap_rounds

        for round_num in range(1, total_rounds + 1):
            is_bootstrap_round = round_num > 1

            if is_bootstrap_round:
                print(f"\n--- Bootstrap round {round_num - 1}/{args.bootstrap_rounds}: "
                      f"evaluating all positions ---", file=sys.stderr)
                evals = evaluate_positions(model, board_t, hand_t, side_t,
                                           own_atk_t, opp_atk_t, torch_mod)
                value_t, pw_t = compute_bootstrap_labels(
                    evals, game_result_t, game_slices, torch_mod,
                    blend=args.bootstrap_blend, drop_margin=args.drop_margin)
                del evals
            else:
                pw_t = None

            print(f"\n=== Round {round_num}/{total_rounds} "
                  f"{'(bootstrap)' if is_bootstrap_round else '(baseline)'} ===",
                  file=sys.stderr)
            final_vloss, final_ploss = train_model(
                model, board_t, hand_t, side_t, own_atk_t, opp_atk_t,
                value_t, policy_t, torch_mod, nn, device,
                args.epochs, args.batch_size, effective_lr,
                resumed=is_bootstrap_round, round_number=round_num,
                policy_weight_t=pw_t, patience=args.patience)
    else:
        # Train (original behavior)
        final_vloss, final_ploss = train_model(
            model, board_t, hand_t, side_t, own_atk_t, opp_atk_t, value_t, policy_t,
            torch_mod, nn, device, args.epochs, args.batch_size, effective_lr,
            resumed=resumed, round_number=args.round,
            patience=args.patience)

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

    # Export to ONNX
    onnx_path = args.output
    print(f"Exporting ONNX to {onnx_path}...", file=sys.stderr)
    save_model = save_model.cpu()
    save_model.eval()
    save_model.freeze_rel_bias()
    batch_size = 1
    board_dummy = torch_mod.zeros(batch_size, 81, dtype=torch_mod.long)
    hand_dummy = torch_mod.zeros(batch_size, 14, dtype=torch_mod.long)
    side_dummy = torch_mod.zeros(batch_size, dtype=torch_mod.long)
    own_atk_dummy = torch_mod.zeros(batch_size, 81, dtype=torch_mod.long)
    opp_atk_dummy = torch_mod.zeros(batch_size, 81, dtype=torch_mod.long)
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
        torch_mod.onnx.export(
            save_model,
            (board_dummy, hand_dummy, side_dummy, own_atk_dummy, opp_atk_dummy),
            onnx_path,
            dynamo=False,
            opset_version=17,
            input_names=["board_tokens", "hand_tokens", "side_token", "own_atk", "opp_atk"],
            output_names=["value", "policy_logits"],
            dynamic_axes={
                "board_tokens": {0: "batch"}, "hand_tokens": {0: "batch"},
                "side_token": {0: "batch"}, "own_atk": {0: "batch"},
                "opp_atk": {0: "batch"}, "value": {0: "batch"},
                "policy_logits": {0: "batch"},
            },
        )
        print(f"ONNX exported: {onnx_path}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}", file=sys.stderr)
        print("Run manually: python tools/export_onnx.py --model " + str(model_path), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
