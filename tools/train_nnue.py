#!/usr/bin/env python3
"""Train the NNUE evaluation network from Floodgate kifu.

Produces a binary nnue.bin file that kishi-to-nnue loads at startup.

The training signal is win/loss outcome: each position is labeled +1 or -1
depending on whether the side to move eventually won. The network learns
to predict the game outcome as a score.

Usage:
  python tools/train_nnue.py --kifu kifu/floodgate
  python tools/train_nnue.py --kifu kifu/floodgate --epochs 10 --lr 1e-3 --batch-size 4096
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
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
BOARD_SIZE = 81
BOARD_FEATURES = 2 * NUM_PIECE_TYPES * BOARD_SIZE   # 2268
HAND_MAX = {1: 18, 2: 4, 3: 4, 4: 4, 5: 4, 6: 2, 7: 2}  # PieceType -> max count
HAND_TYPE_OFFSET = {1: 0, 2: 18, 3: 22, 4: 26, 5: 30, 6: 34, 7: 36}
HAND_FEATURES_PER_COLOR = 38
HAND_FEATURES = 2 * HAND_FEATURES_PER_COLOR  # 76
INPUT_DIM = BOARD_FEATURES + HAND_FEATURES   # 2344

L0_SIZE = 256
L1_SIZE = 32
L2_SIZE = 32

# C++ piece type mapping
PT_PAWN = 1; PT_LANCE = 2; PT_KNIGHT = 3; PT_SILVER = 4
PT_GOLD = 5; PT_BISHOP = 6; PT_ROOK = 7; PT_KING = 8
PT_PROPAWN = 9; PT_PROLANCE = 10; PT_PROKNIGHT = 11; PT_PROSILVER = 12
PT_HORSE = 13; PT_DRAGON = 14


def board_feature_index(perspective_is_black: bool, piece_is_black: bool,
                        piece_type: int, square: int) -> int:
    is_own = (piece_is_black == perspective_is_black)
    color_offset = 0 if is_own else NUM_PIECE_TYPES * BOARD_SIZE
    type_idx = piece_type - 1
    return color_offset + type_idx * BOARD_SIZE + square


def hand_feature_index(perspective_is_black: bool, hand_is_black: bool,
                       piece_type: int, count: int) -> int:
    if count <= 0 or piece_type not in HAND_TYPE_OFFSET:
        return -1
    is_own = (hand_is_black == perspective_is_black)
    offset = HAND_TYPE_OFFSET[piece_type]
    color_offset = 0 if is_own else HAND_FEATURES_PER_COLOR
    return BOARD_FEATURES + color_offset + offset + (count - 1)


CSA_TO_PIECE_TYPE = {
    "FU": PT_PAWN, "KY": PT_LANCE, "KE": PT_KNIGHT, "GI": PT_SILVER,
    "KI": PT_GOLD, "KA": PT_BISHOP, "HI": PT_ROOK, "OU": PT_KING,
    "TO": PT_PROPAWN, "NY": PT_PROLANCE, "NK": PT_PROKNIGHT,
    "NG": PT_PROSILVER, "UM": PT_HORSE, "RY": PT_DRAGON,
}


def extract_features_from_board(board: Board, side_black: bool) -> List[int]:
    """Extract active NNUE feature indices from a csa_parser Board."""
    features = []
    for row in range(9):
        for col in range(9):
            sq = row * 9 + col
            cell = board.board[row][col]
            if cell == 0:
                continue
            piece_is_black = cell > 0
            abs_piece = abs(cell)
            pt_id = PIECE_TYPE_FROM_ID.get(abs_piece)
            if pt_id is None:
                continue
            piece_type = CSA_TO_PIECE_TYPE.get(pt_id)
            if piece_type is None:
                continue
            for persp_black in [True, False]:
                fi = board_feature_index(persp_black, piece_is_black, piece_type, sq)
                if 0 <= fi < INPUT_DIM:
                    features.append((persp_black, fi))

    hand_types_csa = ["FU", "KY", "KE", "GI", "KI", "KA", "HI"]
    hand_types_pt = [PT_PAWN, PT_LANCE, PT_KNIGHT, PT_SILVER, PT_GOLD, PT_BISHOP, PT_ROOK]

    for hand_idx_csa, pt in zip(hand_types_csa, hand_types_pt):
        for color_black in [True, False]:
            hand = board.hand_black if color_black else board.hand_white
            hi = HAND_INDEX.get(hand_idx_csa, -1)
            if hi < 0:
                continue
            count = hand[hi]
            for c in range(1, count + 1):
                for persp_black in [True, False]:
                    fi = hand_feature_index(persp_black, color_black, pt, c)
                    if 0 <= fi < INPUT_DIM:
                        features.append((persp_black, fi))

    return features


def features_to_sparse(features: List[Tuple[bool, int]]) -> Tuple[List[int], List[int]]:
    """Split features into black-perspective and white-perspective index lists."""
    black_feats = [fi for (is_black, fi) in features if is_black]
    white_feats = [fi for (is_black, fi) in features if not is_black]
    return black_feats, white_feats


def parse_game_nnue(filepath: Path, min_rate: int = 0,
                    skip_opening: int = 10,
                    sample_rate: float = 0.3) -> Optional[List[dict]]:
    """Parse CSA file, return list of {black_feats, white_feats, side_black, result}."""
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

    rates = []
    for line in lines:
        if line.startswith("'rating:"):
            try:
                parts = line.split(":")
                for p in parts[1:]:
                    for token in p.split():
                        try:
                            rates.append(int(token))
                        except ValueError:
                            pass
            except Exception:
                pass
    if min_rate > 0 and rates:
        if any(r < min_rate for r in rates):
            return None

    board = Board()
    if not is_hirate(lines):
        parsed = parse_csa_initial_board(lines)
        if parsed is None:
            return None
        board = parsed

    samples = []
    move_num = 0
    side_black = True

    for line in lines:
        if not line or line[0] not in ('+', '-'):
            continue
        if line in ('%TORYO', '%CHUDAN', '%SENNICHITE', '%JISHOGI', '%KACHI', '%HIKIWAKE'):
            break

        move_num += 1
        try:
            side_black = line[0] == '+'
            fr = int(line[1]) - 1, int(line[2]) - 1
            to = int(line[3]) - 1, int(line[4]) - 1
            piece_code = line[5:7]

            if move_num > skip_opening and random.random() < sample_rate:
                features = extract_features_from_board(board, side_black)
                black_feats, white_feats = features_to_sparse(features)
                result = 1.0 if black_win else -1.0
                samples.append({
                    "black_feats": black_feats,
                    "white_feats": white_feats,
                    "side_black": side_black,
                    "result": result,
                })

            # Apply move
            if fr == (-1, -1):
                pi = HAND_INDEX.get(piece_code)
                if pi is not None:
                    hand = board.hand_black if side_black else board.hand_white
                    if hand[pi] > 0:
                        hand[pi] -= 1
                    piece_id = CSA_PIECE_MAP.get(piece_code, 0)
                    board.board[to[0]][to[1]] = piece_id if side_black else -piece_id
            else:
                captured = board.board[to[0]][to[1]]
                if captured != 0:
                    abs_cap = abs(captured)
                    cap_type = PIECE_TYPE_FROM_ID.get(abs_cap)
                    if cap_type:
                        demoted = cap_type
                        promote_map = {"TO": "FU", "NY": "KY", "NK": "KE",
                                       "NG": "GI", "UM": "KA", "RY": "HI"}
                        if demoted in promote_map:
                            demoted = promote_map[demoted]
                        hi = HAND_INDEX.get(demoted)
                        if hi is not None:
                            hand = board.hand_black if side_black else board.hand_white
                            hand[hi] += 1
                board.board[fr[0]][fr[1]] = 0
                piece_id = CSA_PIECE_MAP.get(piece_code, 0)
                board.board[to[0]][to[1]] = piece_id if side_black else -piece_id
        except Exception:
            continue

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
        black_vec = np.zeros(INPUT_DIM, dtype=np.float32)
        white_vec = np.zeros(INPUT_DIM, dtype=np.float32)
        for fi in s["black_feats"]:
            if 0 <= fi < INPUT_DIM:
                black_vec[fi] = 1.0
        for fi in s["white_feats"]:
            if 0 <= fi < INPUT_DIM:
                white_vec[fi] = 1.0
        side = 1.0 if s["side_black"] else -1.0
        result = s["result"] * side
        return black_vec, white_vec, np.float32(side), np.float32(result)


class NNUEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(INPUT_DIM, L0_SIZE)
        self.l1 = nn.Linear(2 * L0_SIZE, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, 1)

    def forward(self, black_input, white_input, side):
        acc_black = torch.clamp(self.l0(black_input), 0.0, 1.0)
        acc_white = torch.clamp(self.l0(white_input), 0.0, 1.0)
        is_black = (side > 0).unsqueeze(1).float()
        own = acc_black * is_black + acc_white * (1.0 - is_black)
        opp = acc_white * is_black + acc_black * (1.0 - is_black)
        concat = torch.cat([own, opp], dim=1)
        h1 = torch.clamp(self.l1(concat), 0.0, 1.0)
        h2 = torch.clamp(self.l2(h1), 0.0, 1.0)
        return self.l3(h2).squeeze(1)


def export_nnue_bin(model: NNUEModel, path: str):
    """Export trained model to binary format matching nnue.cpp's load()."""
    with open(path, "wb") as f:
        f.write(b"NNUE")
        # l0Weights_[INPUT_DIM][L0_SIZE] - row major
        w0 = model.l0.weight.data.t().cpu().numpy()  # [INPUT_DIM, L0_SIZE]
        f.write(w0.astype(np.float32).tobytes())
        f.write(model.l0.bias.data.cpu().numpy().astype(np.float32).tobytes())
        # l1Weights_[2*L0_SIZE][L1_SIZE]
        w1 = model.l1.weight.data.t().cpu().numpy()
        f.write(w1.astype(np.float32).tobytes())
        f.write(model.l1.bias.data.cpu().numpy().astype(np.float32).tobytes())
        # l2Weights_[L1_SIZE][L2_SIZE]
        w2 = model.l2.weight.data.t().cpu().numpy()
        f.write(w2.astype(np.float32).tobytes())
        f.write(model.l2.bias.data.cpu().numpy().astype(np.float32).tobytes())
        # l3Weights_[L2_SIZE]
        f.write(model.l3.weight.data.squeeze(0).cpu().numpy().astype(np.float32).tobytes())
        f.write(model.l3.bias.data.cpu().numpy().astype(np.float32).tobytes())
    print(f"Exported NNUE weights to {path}")


def load_index(kifu_dir: str) -> list:
    """Load or build index of CSA kifu files."""
    kifu_path = Path(kifu_dir)
    files = sorted(kifu_path.rglob("*.csa"))
    return files


def main():
    parser = argparse.ArgumentParser(description="Train NNUE evaluation from Floodgate kifu")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--output", default="nnue.bin", help="Output binary weights (default: nnue.bin)")
    parser.add_argument("--model-pt", default="nnue_model.pt", help="PyTorch checkpoint (default: nnue_model.pt)")
    parser.add_argument("--min-rate", type=int, default=1500, help="Minimum player rating (default: 1500)")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--skip-opening", type=int, default=10, help="Skip first N half-moves (default: 10)")
    parser.add_argument("--sample-rate", type=float, default=0.3, help="Position sampling rate (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel parsing workers (0=auto)")
    parser.add_argument("--resume", default="", help="Resume from PyTorch checkpoint")
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
    train_data = NNUEDataset(all_samples[:split])
    val_data = NNUEDataset(all_samples[split:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NNUEModel().to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=min(4, workers), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=min(4, workers), pin_memory=device.type == "cuda")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for black_vec, white_vec, side, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            black_vec = black_vec.to(device)
            white_vec = white_vec.to(device)
            side = side.to(device)
            target = target.to(device)

            pred = torch.tanh(model(black_vec, white_vec, side))
            loss = loss_fn(pred, target)

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
            for black_vec, white_vec, side, target in val_loader:
                black_vec = black_vec.to(device)
                white_vec = white_vec.to(device)
                side = side.to(device)
                target = target.to(device)
                pred = torch.tanh(model(black_vec, white_vec, side))
                val_loss += loss_fn(pred, target).item() * len(target)
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
