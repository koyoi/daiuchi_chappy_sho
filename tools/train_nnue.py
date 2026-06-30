#!/usr/bin/env python3
"""Train the NNUE evaluation network from Floodgate kifu.

Produces a binary nnue.bin file that kishi-to-nnue loads at startup.

Uses HalfKP features: kingSquare(81) x coloredPieceType(26) x pieceSquare(81)
plus hand piece features (76 dims). Total input = 170662 sparse binary features.

Supports:
- Sigmoid + cross-entropy loss (default) or MSE + tanh (legacy)
- Eval bootstrapping: blend engine eval with game outcome
- Mixed precision training (AMP) for GPU acceleration
- EmbeddingBag for GPU-optimized sparse feature lookup

Usage:
  python tools/train_nnue.py --kifu kifu/floodgate
  python tools/train_nnue.py --kifu kifu/floodgate --epochs 20 --batch-size 65536
  python tools/train_nnue.py --kifu kifu/floodgate --bootstrap nnue_model.pt --lambda-blend 0.5
"""

from __future__ import annotations

import argparse
import os
import random
import re
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is required: pip install torch", file=sys.stderr)
    sys.exit(1)

from csa_parser import (
    Board, CSA_PIECE_MAP, HAND_INDEX, PIECE_TYPE_FROM_ID,
    idx, is_hirate, parse_csa_initial_board,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

# --- NNUE feature constants (must match nnue.h) ---
NUM_PIECE_TYPES = 14
NUM_NON_KING_TYPES = 13
NUM_COLORED_PIECE_TYPES = 2 * NUM_NON_KING_TYPES  # 26
BOARD_SIZE = 81

# HalfKP: kingSquare(81) x coloredPieceType(26) x pieceSquare(81)
BOARD_FEATURES = BOARD_SIZE * NUM_COLORED_PIECE_TYPES * BOARD_SIZE  # 170586

HAND_MAX = {1: 18, 2: 4, 3: 4, 4: 4, 5: 4, 6: 2, 7: 2}
HAND_TYPE_OFFSET = {1: 0, 2: 18, 3: 22, 4: 26, 5: 30, 6: 34, 7: 36}
HAND_FEATURES_PER_COLOR = 38
HAND_FEATURES = 2 * HAND_FEATURES_PER_COLOR  # 76
INPUT_DIM = BOARD_FEATURES + HAND_FEATURES   # 170662

L0_SIZE = 512
WEIGHT_SCALE = 64
EVAL_SCALE = 1.0  # model output is logit of win probability; no rescaling needed

# C++ piece type mapping
PT_PAWN = 1; PT_LANCE = 2; PT_KNIGHT = 3; PT_SILVER = 4
PT_GOLD = 5; PT_BISHOP = 6; PT_ROOK = 7; PT_KING = 8
PT_PROPAWN = 9; PT_PROLANCE = 10; PT_PROKNIGHT = 11; PT_PROSILVER = 12
PT_HORSE = 13; PT_DRAGON = 14


def _piece_type_index(is_own: bool, piece_type: int) -> int:
    if piece_type == PT_KING:
        return -1
    type_idx = piece_type - 1
    if type_idx >= 7:
        type_idx -= 1
    color_offset = 0 if is_own else NUM_NON_KING_TYPES
    return color_offset + type_idx


def board_feature_index(king_square: int, is_own_piece: bool,
                        piece_type: int, piece_square: int) -> int:
    pt_idx = _piece_type_index(is_own_piece, piece_type)
    if pt_idx < 0:
        return -1
    return king_square * NUM_COLORED_PIECE_TYPES * BOARD_SIZE + pt_idx * BOARD_SIZE + piece_square


def hand_feature_index(is_own_hand: bool, piece_type: int, count: int) -> int:
    if count <= 0 or piece_type not in HAND_TYPE_OFFSET:
        return -1
    offset = HAND_TYPE_OFFSET[piece_type]
    color_offset = 0 if is_own_hand else HAND_FEATURES_PER_COLOR
    return BOARD_FEATURES + color_offset + offset + (count - 1)


CSA_TO_PIECE_TYPE = {
    "FU": PT_PAWN, "KY": PT_LANCE, "KE": PT_KNIGHT, "GI": PT_SILVER,
    "KI": PT_GOLD, "KA": PT_BISHOP, "HI": PT_ROOK, "OU": PT_KING,
    "TO": PT_PROPAWN, "NY": PT_PROLANCE, "NK": PT_PROKNIGHT,
    "NG": PT_PROSILVER, "UM": PT_HORSE, "RY": PT_DRAGON,
}


def _find_king_square(board: Board, side: int) -> int:
    king_piece = PT_KING * side
    for sq in range(BOARD_SIZE):
        if board.squares[sq] == king_piece:
            return sq
    return -1


def _is_in_check(board: Board, side: int) -> bool:
    """Simplified check detection for filtering training positions."""
    king_sq = _find_king_square(board, side)
    if king_sq < 0:
        return True
    king_file = (king_sq % 9) + 1
    king_rank = (king_sq // 9) + 1
    opp = -side
    for sq in range(BOARD_SIZE):
        piece = board.squares[sq]
        if piece == 0 or (piece > 0) != (opp > 0):
            continue
        pt = abs(piece)
        pf = (sq % 9) + 1
        pr = (sq // 9) + 1
        df = king_file - pf
        dr = king_rank - pr
        if pt == PT_PAWN and df == 0 and dr == opp:
            return True
        if pt == PT_KNIGHT and abs(df) == 1 and dr == 2 * opp:
            return True
        if pt in (PT_GOLD, PT_PROPAWN, PT_PROLANCE, PT_PROKNIGHT, PT_PROSILVER):
            if abs(df) <= 1 and abs(dr) <= 1:
                if not (dr == opp and abs(df) == 1):
                    pass
                if abs(df) + abs(dr) <= 2 and abs(df) <= 1 and abs(dr) <= 1:
                    if not (dr == -opp and abs(df) == 1):
                        return True
        if pt == PT_SILVER:
            if abs(df) <= 1 and abs(dr) <= 1 and abs(df) + abs(dr) > 0:
                if (dr == opp) or (abs(df) == 1 and dr == -opp):
                    return True
        if pt == PT_KING:
            if abs(df) <= 1 and abs(dr) <= 1 and abs(df) + abs(dr) > 0:
                return True
    return False


def extract_features_from_board(board: Board) -> Tuple[List[int], List[int]]:
    black_king_sq = _find_king_square(board, 1)
    white_king_sq = _find_king_square(board, -1)
    black_feats = []
    white_feats = []

    if black_king_sq < 0 or white_king_sq < 0:
        return black_feats, white_feats

    for sq in range(BOARD_SIZE):
        piece = board.squares[sq]
        if piece == 0:
            continue
        piece_type = abs(piece)
        if piece_type == PT_KING:
            continue
        piece_is_black = piece > 0

        is_own_b = piece_is_black
        fi = board_feature_index(black_king_sq, is_own_b, piece_type, sq)
        if 0 <= fi < INPUT_DIM:
            black_feats.append(fi)

        is_own_w = not piece_is_black
        fi = board_feature_index(white_king_sq, is_own_w, piece_type, sq)
        if 0 <= fi < INPUT_DIM:
            white_feats.append(fi)

    for pt in range(1, 8):
        hi = HAND_INDEX.get(pt)
        if hi is None:
            continue
        for color_black, hand_arr in [(True, board.black_hand), (False, board.white_hand)]:
            count = hand_arr[hi]
            for c in range(1, count + 1):
                fi = hand_feature_index(color_black, pt, c)
                if 0 <= fi < INPUT_DIM:
                    black_feats.append(fi)
                fi = hand_feature_index(not color_black, pt, c)
                if 0 <= fi < INPUT_DIM:
                    white_feats.append(fi)

    return black_feats, white_feats


def parse_game_nnue(filepath: Path, min_rate: int = 0,
                    skip_opening: int = 0,
                    sample_rate: float = 1.0,
                    skip_in_check: bool = True,
                    discount_halflife: float = 60.0) -> Optional[List[dict]]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    lines = [l.rstrip("\n\r") for l in lines]

    result_line = None
    for line in lines:
        if line.startswith("'summary:"):
            result_line = line
            break
    if not result_line:
        return None

    black_win = None
    if "lose" in result_line and "win" in result_line:
        parts = result_line.split(":")
        if len(parts) >= 4:
            p1_result = parts[2].split()[-1] if parts[2] else ""
            p2_result = parts[3].split()[-1] if len(parts) > 3 and parts[3] else ""
            if p1_result == "win":
                black_win = True
            elif p2_result == "win":
                black_win = False
            elif p1_result == "lose":
                black_win = False
            elif p2_result == "lose":
                black_win = True
    if black_win is None:
        return None

    if min_rate > 0:
        black_rate = 0
        white_rate = 0
        for line in lines:
            if line.startswith("'black_rate:"):
                m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
                if m:
                    black_rate = int(float(m.group(1)))
            elif line.startswith("'white_rate:"):
                m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
                if m:
                    white_rate = int(float(m.group(1)))
        if black_rate < min_rate or white_rate < min_rate:
            return None

    board_setup_lines = [l for l in lines if re.match(r"^P[1-9+\-]", l)]
    if is_hirate(lines):
        std_lines = [
            "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY",
            "P2 * -HI *  *  *  *  * -KA * ",
            "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU",
            "P4 *  *  *  *  *  *  *  *  * ",
            "P5 *  *  *  *  *  *  *  *  * ",
            "P6 *  *  *  *  *  *  *  *  * ",
            "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU",
            "P8 * +KA *  *  *  *  * +HI * ",
            "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY",
        ]
        board = parse_csa_initial_board(std_lines)
    else:
        board = parse_csa_initial_board(board_setup_lines)

    for line in lines:
        if line == "+":
            board.side = 1
            break
        elif line == "-":
            board.side = -1
            break

    move_lines = []
    for line in lines:
        if len(line) >= 7 and line[0] in ('+', '-') and line[1].isdigit():
            move_lines.append(line)

    if len(move_lines) < 10:
        return None

    total_plies = len(move_lines)
    samples = []
    for ply, mline in enumerate(move_lines):
        side = 1 if mline[0] == '+' else -1
        side_black = (side == 1)

        from_file = int(mline[1])
        from_rank = int(mline[2])
        to_file = int(mline[3])
        to_rank = int(mline[4])
        piece_name = mline[5:7]
        pt = CSA_PIECE_MAP.get(piece_name, 0)
        if pt == 0:
            continue

        is_drop = (from_file == 0 and from_rank == 0)
        to_sq = idx(to_file, to_rank)

        if is_drop:
            from_sq = -1
            promote = False
            drop_piece = pt
        else:
            from_sq = idx(from_file, from_rank)
            drop_piece = 0
            current_piece = abs(board.squares[from_sq]) if from_sq >= 0 else 0
            promote = (pt >= 9 and current_piece < 9) if current_piece > 0 else False

        include_prob = min(1.0, ply / 30.0) * sample_rate
        if ply >= skip_opening and random.random() < include_prob:
            if not (skip_in_check and _is_in_check(board, side)):
                bf, wf = extract_features_from_board(board)
                plies_remaining = total_plies - ply
                discount = 0.5 ** (plies_remaining / discount_halflife)
                win_prob = 1.0 if (black_win == side_black) else 0.0
                label = 0.5 + discount * (win_prob - 0.5)
                samples.append({
                    "black_feats": np.array(bf, dtype=np.int32),
                    "white_feats": np.array(wf, dtype=np.int32),
                    "side_black": side_black,
                    "soft_label": label,
                })

        board.apply_move(from_sq, to_sq, is_drop,
                         drop_piece if is_drop else pt,
                         promote, side)

    return samples if samples else None


def _parse_worker(args_tuple):
    filepath, min_rate, skip_opening, sample_rate, discount_halflife = args_tuple
    return parse_game_nnue(Path(filepath), min_rate=min_rate,
                           skip_opening=skip_opening,
                           sample_rate=sample_rate,
                           discount_halflife=discount_halflife)


class NNUEDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        side = 1.0 if s["side_black"] else -1.0
        result = s["result"] * side
        return s["black_feats"], s["white_feats"], np.float32(side), np.float32(result)


class NNUEDatasetWithSoftLabels(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        side = 1.0 if s["side_black"] else -1.0
        if "soft_label" in s:
            label = s["soft_label"]
        else:
            result = s["result"] * side
            label = (result + 1.0) / 2.0
        return s["black_feats"], s["white_feats"], np.float32(side), np.float32(label)


def _pack_indices(index_list):
    """Pack per-sample index arrays into EmbeddingBag format (flat indices + offsets)."""
    lengths = np.array([len(x) for x in index_list], dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return torch.zeros(1, dtype=torch.long), torch.zeros(len(index_list), dtype=torch.long)
    indices = np.concatenate([np.asarray(x, dtype=np.int64) for x in index_list])
    offsets = np.empty(len(index_list), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])
    return torch.from_numpy(indices), torch.from_numpy(offsets)


def collate_embag(batch):
    """Collate function producing EmbeddingBag-format packed indices + offsets."""
    black_list, white_list, sides, targets = zip(*batch)
    b_idx, b_off = _pack_indices(black_list)
    w_idx, w_off = _pack_indices(white_list)
    sides_t = torch.tensor(np.array(sides), dtype=torch.float32)
    targets_t = torch.tensor(np.array(targets), dtype=torch.float32)
    return b_idx, b_off, w_idx, w_off, sides_t, targets_t


class NNUEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.EmbeddingBag(INPUT_DIM, L0_SIZE, mode='sum', sparse=False)
        self.l0_bias = nn.Parameter(torch.full((L0_SIZE,), 0.5))
        self.out_weight = nn.Parameter(torch.zeros(2 * L0_SIZE))
        self.out_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.l0.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out_weight, mean=0.0, std=1.0 / (2 * L0_SIZE) ** 0.5)
        nn.init.zeros_(self.out_bias)

    def forward(self, b_idx, b_off, w_idx, w_off, side):
        acc_black = torch.clamp(self.l0(b_idx, b_off) + self.l0_bias, 0.0, 1.0) ** 2
        acc_white = torch.clamp(self.l0(w_idx, w_off) + self.l0_bias, 0.0, 1.0) ** 2
        is_black = (side > 0).unsqueeze(1).float()
        own = acc_black * is_black + acc_white * (1.0 - is_black)
        opp = acc_white * is_black + acc_black * (1.0 - is_black)
        concat = torch.cat([own, opp], dim=1)
        return (concat * self.out_weight).sum(dim=1) + self.out_bias


def sigmoid_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


def bootstrap_labels(model, dataset, device, lambda_blend=0.5, batch_size=65536):
    """Replace binary game outcomes with blended labels using model predictions."""
    model.eval()
    dl_workers = 0 if sys.platform == "win32" else min(8, os.cpu_count() or 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_embag, num_workers=dl_workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=dl_workers > 0)
    soft_labels = []
    with torch.no_grad():
        for b_idx, b_off, w_idx, w_off, side, target in tqdm(loader, desc="Bootstrapping"):
            b_idx, b_off = b_idx.to(device, non_blocking=True), b_off.to(device, non_blocking=True)
            w_idx, w_off = w_idx.to(device, non_blocking=True), w_off.to(device, non_blocking=True)
            side_d = side.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw_pred = model(b_idx, b_off, w_idx, w_off, side_d)
            engine_prob = torch.sigmoid(raw_pred.float())
            blended = lambda_blend * engine_prob.cpu() + (1.0 - lambda_blend) * target
            soft_labels.append(blended)

    soft_labels = torch.cat(soft_labels).numpy()
    for i in range(len(dataset.samples)):
        dataset.samples[i]["soft_label"] = float(soft_labels[i])
    return dataset


def export_nnue_bin(model_or_module: nn.Module, path: str):
    """Export trained model to NNU5 binary format (2xL0 -> 1)."""
    model = model_or_module.module if isinstance(model_or_module, nn.DataParallel) else model_or_module

    with open(path, "wb") as f:
        f.write(b"NNU5")
        f.write(struct.pack('<i', L0_SIZE))
        w0 = model.l0.weight.data.cpu()
        w0_q = (w0 * WEIGHT_SCALE).round().clamp(-32768, 32767).to(torch.int16)
        f.write(w0_q.numpy().tobytes())
        b0 = model.l0_bias.data.cpu()
        b0_q = (b0 * WEIGHT_SCALE).round().clamp(-2147483648, 2147483647).to(torch.int32)
        f.write(b0_q.numpy().tobytes())
        f.write(model.out_weight.data.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.out_bias.data.cpu().numpy().astype(np.float32).tobytes())
    print(f"Exported NNU5 weights to {path}")


def _load_state_dict_compat(model, state_dict):
    """Load state dict with backward compatibility."""
    if 'l0.weight' in state_dict and state_dict['l0.weight'].shape == (L0_SIZE, INPUT_DIM):
        state_dict['l0.weight'] = state_dict['l0.weight'].t().contiguous()
    if 'l0.bias' in state_dict and 'l0_bias' not in state_dict:
        state_dict['l0_bias'] = state_dict.pop('l0.bias')
    for key in list(state_dict.keys()):
        if key.startswith(('l1.', 'l2.', 'l3.')):
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)


def _cache_meta(args) -> dict:
    """Capture parse parameters so the cache can detect stale settings."""
    return {
        "min_rate": args.min_rate,
        "skip_opening": args.skip_opening,
        "sample_rate": args.sample_rate,
        "discount_halflife": args.discount_halflife,
    }


def _kifu_fingerprint(kifu_dir: str) -> str:
    """Fast fingerprint: file count + total size + newest mtime."""
    kifu_path = Path(kifu_dir)
    count = 0
    total_size = 0
    newest = 0.0
    for f in kifu_path.rglob("*.csa"):
        count += 1
        st = f.stat()
        total_size += st.st_size
        if st.st_mtime > newest:
            newest = st.st_mtime
    return f"{count}_{total_size}_{int(newest)}"


def save_cache(path: str, samples: List[dict], fingerprint: str, meta: dict):
    """Save parsed samples to npz for fast reload."""
    import json
    n = len(samples)
    print(f"Saving cache ({n:,} positions)...")
    t0 = __import__('time').time()

    all_black = np.concatenate([s["black_feats"] for s in samples])
    all_white = np.concatenate([s["white_feats"] for s in samples])
    black_lengths = np.array([len(s["black_feats"]) for s in samples], dtype=np.int32)
    white_lengths = np.array([len(s["white_feats"]) for s in samples], dtype=np.int32)
    side_black = np.array([s["side_black"] for s in samples], dtype=np.bool_)
    soft_label = np.array([s["soft_label"] for s in samples], dtype=np.float32)

    np.savez(path,
             black_feats=all_black, white_feats=all_white,
             black_lengths=black_lengths, white_lengths=white_lengths,
             side_black=side_black, soft_label=soft_label,
             fingerprint=np.array([fingerprint]),
             meta_json=np.array([json.dumps(meta)]))
    elapsed = __import__('time').time() - t0
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.0f} MB, {elapsed:.1f}s)")


def load_cache(path: str, fingerprint: str, meta: dict) -> Optional[List[dict]]:
    """Load cache if it exists and matches current kifu directory + settings."""
    import json
    if not os.path.exists(path):
        return None
    print(f"Loading cache: {path}...", end="", flush=True)
    t0 = __import__('time').time()
    data = np.load(path, allow_pickle=False)

    cached_fp = str(data["fingerprint"][0]) if "fingerprint" in data else ""
    if cached_fp != fingerprint:
        print(f" STALE (kifu changed), reparsing")
        return None
    if "meta_json" in data:
        cached_meta = json.loads(str(data["meta_json"][0]))
        if cached_meta != meta:
            print(f" STALE (settings changed), reparsing")
            return None

    all_black = data["black_feats"]
    all_white = data["white_feats"]
    black_lengths = data["black_lengths"]
    white_lengths = data["white_lengths"]
    side_black_arr = data["side_black"]
    soft_label_arr = data["soft_label"]

    n = len(black_lengths)
    b_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(black_lengths, out=b_offsets[1:])
    w_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(white_lengths, out=w_offsets[1:])

    samples = []
    for i in range(n):
        samples.append({
            "black_feats": all_black[b_offsets[i]:b_offsets[i+1]],
            "white_feats": all_white[w_offsets[i]:w_offsets[i+1]],
            "side_black": bool(side_black_arr[i]),
            "soft_label": float(soft_label_arr[i]),
        })

    elapsed = __import__('time').time() - t0
    print(f" {n:,} positions in {elapsed:.1f}s")
    return samples


def load_index(kifu_dir: str) -> list:
    kifu_path = Path(kifu_dir)
    files = sorted(kifu_path.rglob("*.csa"))
    return files


def load_data_npz(path: str) -> Optional[List[dict]]:
    """Load training data from NPZ file without fingerprint checking.

    Used for self-play data or any pre-generated NPZ from nnue_self_play.py.
    """
    if not os.path.exists(path):
        print(f"Data file not found: {path}", file=sys.stderr)
        return None
    print(f"Loading data: {path}...", end="", flush=True)
    t0 = __import__('time').time()
    data = np.load(path, allow_pickle=False)

    all_black = data["black_feats"]
    all_white = data["white_feats"]
    black_lengths = data["black_lengths"]
    white_lengths = data["white_lengths"]
    side_black_arr = data["side_black"]
    soft_label_arr = data["soft_label"]

    n = len(black_lengths)
    b_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(black_lengths, out=b_offsets[1:])
    w_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(white_lengths, out=w_offsets[1:])

    samples = []
    for i in range(n):
        samples.append({
            "black_feats": all_black[b_offsets[i]:b_offsets[i+1]],
            "white_feats": all_white[w_offsets[i]:w_offsets[i+1]],
            "side_black": bool(side_black_arr[i]),
            "soft_label": float(soft_label_arr[i]),
        })

    elapsed = __import__('time').time() - t0
    print(f" {n:,} positions in {elapsed:.1f}s")

    if "meta_json" in data:
        import json
        meta = json.loads(str(data["meta_json"][0]))
        print(f"  Source: {meta.get('source', 'unknown')}")
        if "teacher" in meta:
            print(f"  Teacher: {meta['teacher']}, movetime: {meta.get('movetime', '?')}ms")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train NNUE evaluation from Floodgate kifu")
    parser.add_argument("--kifu", default="", help="Kifu directory (optional if --data is used)")
    parser.add_argument("--data", default="", help="Load training data directly from NPZ (e.g. self-play output)")
    parser.add_argument("--cache", default="", help="Cache file (.npz) for parsed data; auto-invalidates on kifu/settings change")
    parser.add_argument("--output", default="nnue.bin", help="Output binary weights (default: nnue.bin)")
    parser.add_argument("--model-pt", default="nnue_model.pt", help="PyTorch checkpoint (default: nnue_model.pt)")
    parser.add_argument("--min-rate", type=int, default=2300, help="Minimum player rating (default: 2300)")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--skip-opening", type=int, default=0, help="Skip first N half-moves (default: 0; opening ramp-up handles this)")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Position sampling rate (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=65536, help="Batch size (default: 65536)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel parsing workers (0=auto)")
    parser.add_argument("--resume", default="", help="Resume from PyTorch checkpoint")
    parser.add_argument("--bootstrap", default="", help="Path to existing model checkpoint for eval bootstrapping")
    parser.add_argument("--lambda-blend", type=float, default=0.5,
                        help="Bootstrap blend: lambda*engine + (1-lambda)*outcome (default: 0.5)")
    parser.add_argument("--discount-halflife", type=float, default=60.0,
                        help="Result discount half-life in plies (default: 60)")
    parser.add_argument("--loss", choices=["sigmoid", "mse"], default="sigmoid",
                        help="Loss function: sigmoid (default) or mse (legacy)")
    args = parser.parse_args()

    if args.data:
        all_samples = load_data_npz(args.data)
        if not all_samples:
            return 1
        print(f"Total training samples: {len(all_samples):,}")
    elif args.kifu:
        print("Loading kifu index...")
        files = load_index(args.kifu)
        if not files:
            print(f"No .csa files found in {args.kifu}", file=sys.stderr)
            return 1
        print(f"Found {len(files)} kifu files")

        fingerprint = _kifu_fingerprint(args.kifu)
        meta = _cache_meta(args)
        cache_path = args.cache if args.cache else ""
        if not cache_path:
            cache_path = os.path.join(args.kifu, "nnue_cache.npz")

        all_samples = load_cache(cache_path, fingerprint, meta)

        if all_samples is None:
            if args.max_games > 0:
                random.shuffle(files)
                files = files[:args.max_games]

            print("Parsing kifu files...")
            workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 16)
            tasks = [(str(f), args.min_rate, args.skip_opening, args.sample_rate, args.discount_halflife) for f in files]

            all_samples = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in tqdm(pool.map(_parse_worker, tasks, chunksize=64), total=len(tasks), desc="Parsing"):
                    if result:
                        all_samples.extend(result)

            if not all_samples:
                print("No training samples extracted", file=sys.stderr)
                return 1

            print(f"Total training samples: {len(all_samples):,}")
            save_cache(cache_path, all_samples, fingerprint, meta)
        else:
            print(f"Total training samples: {len(all_samples):,} (from cache)")
    else:
        print("Either --kifu or --data is required", file=sys.stderr)
        return 1

    random.shuffle(all_samples)

    split = int(len(all_samples) * 0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"Device: {device} (GPUs: {num_gpus})")

    model = NNUEModel().to(device)
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        _load_state_dict_compat(model, sd)
        print(f"Resumed from {args.resume}")

    # Eval bootstrapping
    use_sigmoid = (args.loss == "sigmoid")
    if args.bootstrap and os.path.exists(args.bootstrap):
        boot_model = NNUEModel().to(device)
        sd = torch.load(args.bootstrap, map_location=device, weights_only=False)
        _load_state_dict_compat(boot_model, sd)
        print(f"Bootstrapping with model: {args.bootstrap} (lambda={args.lambda_blend})")
        boot_dataset = NNUEDatasetWithSoftLabels(all_samples)
        boot_dataset = bootstrap_labels(boot_model, boot_dataset, device,
                                         lambda_blend=args.lambda_blend)
        train_data = NNUEDatasetWithSoftLabels(boot_dataset.samples[:split])
        val_data = NNUEDatasetWithSoftLabels(boot_dataset.samples[split:])
        del boot_model
        use_sigmoid = True
    else:
        train_data = NNUEDatasetWithSoftLabels(all_samples[:split])
        val_data = NNUEDatasetWithSoftLabels(all_samples[split:])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dl_workers = 0 if sys.platform == "win32" else min(8, args.workers if args.workers > 0 else (os.cpu_count() or 4))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=dl_workers, pin_memory=device.type == "cuda",
                              collate_fn=collate_embag, persistent_workers=dl_workers > 0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=dl_workers, pin_memory=device.type == "cuda",
                            collate_fn=collate_embag, persistent_workers=dl_workers > 0)

    if use_sigmoid:
        print("Loss: sigmoid cross-entropy (no rescaling)")
    else:
        print("Loss: MSE + tanh (legacy)")
    loss_fn_mse = nn.MSELoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Using mixed precision training (AMP)")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for b_idx, b_off, w_idx, w_off, side, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            b_idx, b_off = b_idx.to(device), b_off.to(device)
            w_idx, w_off = w_idx.to(device), w_off.to(device)
            side = side.to(device)
            target = target.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(b_idx, b_off, w_idx, w_off, side)
                if use_sigmoid:
                    loss = sigmoid_loss(pred, target)
                else:
                    loss = loss_fn_mse(torch.tanh(pred), target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(target)
            train_n += len(target)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for b_idx, b_off, w_idx, w_off, side, target in val_loader:
                b_idx, b_off = b_idx.to(device), b_off.to(device)
                w_idx, w_off = w_idx.to(device), w_off.to(device)
                side = side.to(device)
                target = target.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(b_idx, b_off, w_idx, w_off, side)
                    if use_sigmoid:
                        vloss = sigmoid_loss(pred, target)
                    else:
                        vloss = loss_fn_mse(torch.tanh(pred), target)
                val_loss += vloss.item() * len(target)
                val_n += len(target)

        tl = train_loss / max(train_n, 1)
        vl = val_loss / max(val_n, 1)
        print(f"  train_loss={tl:.6f}  val_loss={vl:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), args.model_pt)
            export_nnue_bin(model, args.output)
            print(f"  -> Best model saved (val_loss={vl:.6f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
    print(f"Weights: {args.output}")
    print(f"PyTorch model: {args.model_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
