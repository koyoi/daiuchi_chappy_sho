#!/usr/bin/env python3
"""Train the NNUE evaluation network from Floodgate kifu.

Produces a binary nnue.bin file that kishi-to-nnue loads at startup.

Uses HalfKP features: kingSquare(81) x coloredPieceType(26) x pieceSquare(81)
plus hand piece features (76 dims). Total input = 170662 sparse binary features.

Supports:
- Sigmoid + cross-entropy loss (default) or MSE + tanh (legacy)
- Eval bootstrapping: blend engine eval with game outcome
- Multi-GPU training via DataParallel (e.g. RTX PRO 6000 x2)

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
L1_SIZE = 32
L2_SIZE = 32
WEIGHT_SCALE = 64
EVAL_SCALE = 361.0  # sigmoid scaling factor (Stockfish convention)

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
                    skip_opening: int = 24,
                    sample_rate: float = 0.3,
                    skip_in_check: bool = True) -> Optional[List[dict]]:
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

        if ply >= skip_opening and random.random() < sample_rate:
            if not (skip_in_check and _is_in_check(board, side)):
                bf, wf = extract_features_from_board(board)
                result = 1.0 if black_win else -1.0
                samples.append({
                    "black_feats": bf,
                    "white_feats": wf,
                    "side_black": side_black,
                    "result": result,
                })

        board.apply_move(from_sq, to_sq, is_drop,
                         drop_piece if is_drop else pt,
                         promote, side)

    return samples if samples else None


def _parse_worker(args_tuple):
    filepath, min_rate, skip_opening, sample_rate = args_tuple
    return parse_game_nnue(Path(filepath), min_rate=min_rate,
                           skip_opening=skip_opening,
                           sample_rate=sample_rate)


class NNUEDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        black_indices = [fi for fi in s["black_feats"] if 0 <= fi < INPUT_DIM]
        white_indices = [fi for fi in s["white_feats"] if 0 <= fi < INPUT_DIM]
        side = 1.0 if s["side_black"] else -1.0
        result = s["result"] * side  # +1 = side-to-move wins, -1 = loses
        return black_indices, white_indices, np.float32(side), np.float32(result)


def collate_sparse(batch):
    black_indices_list, white_indices_list, sides, targets = zip(*batch)
    batch_size = len(batch)

    black_rows, black_cols = [], []
    white_rows, white_cols = [], []
    for i in range(batch_size):
        for fi in black_indices_list[i]:
            black_rows.append(i)
            black_cols.append(fi)
        for fi in white_indices_list[i]:
            white_rows.append(i)
            white_cols.append(fi)

    if black_rows:
        black_idx = torch.LongTensor([black_rows, black_cols])
        black_val = torch.ones(len(black_rows))
        black_sparse = torch.sparse_coo_tensor(black_idx, black_val, (batch_size, INPUT_DIM))
    else:
        black_sparse = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long),
                                                torch.zeros(0), (batch_size, INPUT_DIM))

    if white_rows:
        white_idx = torch.LongTensor([white_rows, white_cols])
        white_val = torch.ones(len(white_rows))
        white_sparse = torch.sparse_coo_tensor(white_idx, white_val, (batch_size, INPUT_DIM))
    else:
        white_sparse = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long),
                                                torch.zeros(0), (batch_size, INPUT_DIM))

    sides_t = torch.stack([torch.tensor(s) for s in sides])
    targets_t = torch.stack([torch.tensor(t) for t in targets])
    return black_sparse, white_sparse, sides_t, targets_t


class NNUEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(INPUT_DIM, L0_SIZE)
        self.l1 = nn.Linear(2 * L0_SIZE, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, 1)

    def forward(self, black_input, white_input, side):
        # SCReLU: clamp(x, 0, 1)^2
        acc_black = torch.clamp(self.l0(black_input), 0.0, 1.0) ** 2
        acc_white = torch.clamp(self.l0(white_input), 0.0, 1.0) ** 2
        is_black = (side > 0).unsqueeze(1).float()
        own = acc_black * is_black + acc_white * (1.0 - is_black)
        opp = acc_white * is_black + acc_black * (1.0 - is_black)
        concat = torch.cat([own, opp], dim=1)
        h1 = torch.clamp(self.l1(concat), 0.0, 1.0)
        h2 = torch.clamp(self.l2(h1), 0.0, 1.0)
        return self.l3(h2).squeeze(1)


def sigmoid_loss(pred, target, scale=EVAL_SCALE):
    """Sigmoid cross-entropy loss.

    pred: raw network output (before sigmoid)
    target: probability of side-to-move winning (0.0 or 1.0, or soft label 0..1)
    """
    scaled = pred / scale
    return F.binary_cross_entropy_with_logits(scaled, target, reduction='mean')


def bootstrap_labels(model, dataset, device, lambda_blend=0.5, batch_size=8192):
    """Replace binary game outcomes with blended labels using model predictions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_sparse, num_workers=0)
    soft_labels = []
    with torch.no_grad():
        for black_sparse, white_sparse, side, target in tqdm(loader, desc="Bootstrapping"):
            black_vec = black_sparse.to_dense().to(device)
            white_vec = white_sparse.to_dense().to(device)
            side_d = side.to(device)
            raw_pred = model(black_vec, white_vec, side_d)
            engine_prob = torch.sigmoid(raw_pred / EVAL_SCALE)
            game_prob = (target + 1.0) / 2.0  # -1..+1 -> 0..1
            blended = lambda_blend * engine_prob.cpu() + (1.0 - lambda_blend) * game_prob
            soft_labels.append(blended)

    soft_labels = torch.cat(soft_labels).numpy()
    for i, s in enumerate(dataset.samples):
        # Store as probability (0..1) for sigmoid loss
        dataset.samples[i]["soft_label"] = float(soft_labels[i])
    return dataset


class NNUEDatasetWithSoftLabels(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        black_indices = [fi for fi in s["black_feats"] if 0 <= fi < INPUT_DIM]
        white_indices = [fi for fi in s["white_feats"] if 0 <= fi < INPUT_DIM]
        side = 1.0 if s["side_black"] else -1.0
        if "soft_label" in s:
            label = s["soft_label"]
        else:
            result = s["result"] * side
            label = (result + 1.0) / 2.0  # -1..+1 -> 0..1
        return black_indices, white_indices, np.float32(side), np.float32(label)


def export_nnue_bin(model_or_module: nn.Module, path: str):
    """Export trained model to NNU3 binary format.

    Unwraps DataParallel if needed.
    """
    model = model_or_module.module if isinstance(model_or_module, nn.DataParallel) else model_or_module

    with open(path, "wb") as f:
        f.write(b"NNU4")
        f.write(struct.pack('<i', L0_SIZE))
        w0 = model.l0.weight.data.t().cpu()
        w0_q = (w0 * WEIGHT_SCALE).round().clamp(-32768, 32767).to(torch.int16)
        f.write(w0_q.numpy().tobytes())
        b0 = model.l0.bias.data.cpu()
        b0_q = (b0 * WEIGHT_SCALE).round().clamp(-2147483648, 2147483647).to(torch.int32)
        f.write(b0_q.numpy().tobytes())
        w1 = model.l1.weight.data.t().cpu().numpy()
        f.write(w1.astype(np.float32).tobytes())
        f.write(model.l1.bias.data.cpu().numpy().astype(np.float32).tobytes())
        w2 = model.l2.weight.data.t().cpu().numpy()
        f.write(w2.astype(np.float32).tobytes())
        f.write(model.l2.bias.data.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.l3.weight.data.squeeze(0).cpu().numpy().astype(np.float32).tobytes())
        f.write(model.l3.bias.data.cpu().numpy().astype(np.float32).tobytes())
    print(f"Exported NNU4 weights to {path}")


def load_index(kifu_dir: str) -> list:
    kifu_path = Path(kifu_dir)
    files = sorted(kifu_path.rglob("*.csa"))
    return files


def main():
    parser = argparse.ArgumentParser(description="Train NNUE evaluation from Floodgate kifu")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--output", default="nnue.bin", help="Output binary weights (default: nnue.bin)")
    parser.add_argument("--model-pt", default="nnue_model.pt", help="PyTorch checkpoint (default: nnue_model.pt)")
    parser.add_argument("--min-rate", type=int, default=2500, help="Minimum player rating (default: 2500)")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--skip-opening", type=int, default=24, help="Skip first N half-moves (default: 24)")
    parser.add_argument("--sample-rate", type=float, default=0.3, help="Position sampling rate (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=65536, help="Batch size (default: 65536)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel parsing workers (0=auto)")
    parser.add_argument("--resume", default="", help="Resume from PyTorch checkpoint")
    parser.add_argument("--bootstrap", default="", help="Path to existing model checkpoint for eval bootstrapping")
    parser.add_argument("--lambda-blend", type=float, default=0.5,
                        help="Bootstrap blend: lambda*engine + (1-lambda)*outcome (default: 0.5)")
    parser.add_argument("--loss", choices=["sigmoid", "mse"], default="sigmoid",
                        help="Loss function: sigmoid (default) or mse (legacy)")
    args = parser.parse_args()

    print("Loading kifu index...")
    files = load_index(args.kifu)
    if not files:
        print(f"No .csa files found in {args.kifu}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} kifu files")

    if args.max_games > 0:
        random.shuffle(files)
        files = files[:args.max_games]

    print("Parsing kifu files...")
    workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 16)
    tasks = [(str(f), args.min_rate, args.skip_opening, args.sample_rate) for f in files]

    all_samples = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(_parse_worker, tasks, chunksize=64), total=len(tasks), desc="Parsing"):
            if result:
                all_samples.extend(result)

    if not all_samples:
        print("No training samples extracted", file=sys.stderr)
        return 1

    print(f"Total training samples: {len(all_samples)}")
    random.shuffle(all_samples)

    split = int(len(all_samples) * 0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"Device: {device} (GPUs: {num_gpus})")

    model = NNUEModel().to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    # Eval bootstrapping
    use_sigmoid = (args.loss == "sigmoid")
    if args.bootstrap and os.path.exists(args.bootstrap):
        boot_model = NNUEModel().to(device)
        boot_model.load_state_dict(torch.load(args.bootstrap, map_location=device, weights_only=True))
        print(f"Bootstrapping with model: {args.bootstrap} (lambda={args.lambda_blend})")
        boot_dataset = NNUEDatasetWithSoftLabels(all_samples)
        boot_dataset = bootstrap_labels(boot_model, boot_dataset, device,
                                         lambda_blend=args.lambda_blend)
        train_data = NNUEDatasetWithSoftLabels(boot_dataset.samples[:split])
        val_data = NNUEDatasetWithSoftLabels(boot_dataset.samples[split:])
        del boot_model
        use_sigmoid = True
    else:
        if use_sigmoid:
            train_data = NNUEDatasetWithSoftLabels(all_samples[:split])
            val_data = NNUEDatasetWithSoftLabels(all_samples[split:])
        else:
            train_data = NNUEDataset(all_samples[:split])
            val_data = NNUEDataset(all_samples[split:])

    # Multi-GPU
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dl_workers = 0 if sys.platform == "win32" else min(4, workers)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=dl_workers, pin_memory=device.type == "cuda",
                              collate_fn=collate_sparse)
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=dl_workers, pin_memory=device.type == "cuda",
                            collate_fn=collate_sparse)

    if use_sigmoid:
        print(f"Loss: sigmoid cross-entropy (scale={EVAL_SCALE})")
    else:
        print("Loss: MSE + tanh (legacy)")
    loss_fn_mse = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for black_sparse, white_sparse, side, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            black_vec = black_sparse.to_dense().to(device)
            white_vec = white_sparse.to_dense().to(device)
            side = side.to(device)
            target = target.to(device)

            pred = model(black_vec, white_vec, side)
            if use_sigmoid:
                loss = sigmoid_loss(pred, target)
            else:
                loss = loss_fn_mse(torch.tanh(pred), target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(target)
            train_n += len(target)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for black_sparse, white_sparse, side, target in val_loader:
                black_vec = black_sparse.to_dense().to(device)
                white_vec = white_sparse.to_dense().to(device)
                side = side.to(device)
                target = target.to(device)
                pred = model(black_vec, white_vec, side)
                if use_sigmoid:
                    val_loss += sigmoid_loss(pred, target).item() * len(target)
                else:
                    val_loss += loss_fn_mse(torch.tanh(pred), target).item() * len(target)
                val_n += len(target)

        tl = train_loss / max(train_n, 1)
        vl = val_loss / max(val_n, 1)
        print(f"  train_loss={tl:.6f}  val_loss={vl:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw_model.state_dict(), args.model_pt)
            export_nnue_bin(model, args.output)
            print(f"  -> Best model saved (val_loss={vl:.6f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
    print(f"Weights: {args.output}")
    print(f"PyTorch model: {args.model_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
