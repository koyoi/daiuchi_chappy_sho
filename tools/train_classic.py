#!/usr/bin/env python3
"""Train the classic engine's 44-feature evaluation from Floodgate kifu.

Bonanza method: for each position, adjust weights so the engine prefers
the move played by the higher-rated winner.

Usage:
  python tools/train_classic.py --kifu kifu/floodgate --engine build/Release/kishi-to-classic.exe
  python tools/train_classic.py --kifu kifu/floodgate --engine build/Release/kishi-to-classic.exe --epochs 3 --lr 0.01
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from csa_parser import (
    Board, CSA_PIECE_MAP, HAND_INDEX, PIECE_TYPE_FROM_ID,
    idx, is_hirate, parse_csa_initial_board,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def parse_game_for_classic(filepath: Path, min_rate: int = 0,
                           skip_opening: int = 10,
                           sample_rate: float = 1.0) -> Optional[List[dict]]:
    """Parse a CSA kifu file for classic engine training.

    Returns list of {sfen, usi_move} for positions where the higher-rated
    winner moved, skipping the first skip_opening moves.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    lines = [l.rstrip("\n\r") for l in lines]

    # Extract result
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

    # Extract ratings
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

    if min_rate > 0 and (black_rate < min_rate or white_rate < min_rate):
        return None

    # Determine the higher-rated player; require that they also won
    if black_rate >= white_rate:
        teacher_side = 1  # Black
        if not black_win:
            return None
    else:
        teacher_side = -1  # White
        if black_win:
            return None

    # Parse initial position
    board_setup_lines = [l for l in lines if re.match(r"^P[1-9+\-]", l)]
    if is_hirate(board_setup_lines):
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

    # Parse moves
    move_lines = []
    for line in lines:
        if len(line) >= 7 and line[0] in ('+', '-') and line[1].isdigit():
            move_lines.append(line)

    if len(move_lines) < 10:
        return None

    samples = []
    move_number = 1

    for ply, mline in enumerate(move_lines):
        if len(mline) < 7:
            continue
        side = 1 if mline[0] == '+' else -1

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

        # Only learn from the teacher's moves, past the opening
        if side == teacher_side and ply >= skip_opening:
            if sample_rate >= 1.0 or random.random() < sample_rate:
                sfen = board.to_sfen(move_number)
                usi_move = Board.move_to_usi(from_sq, to_sq, is_drop,
                                            drop_piece if is_drop else 0, promote)
                samples.append({"sfen": sfen, "usi_move": usi_move})

        # Apply move
        board.apply_move(from_sq, to_sq, is_drop, drop_piece if is_drop else pt,
                         promote, side)
        if side == -1:
            move_number += 1

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train classic engine evaluation from Floodgate kifu (Bonanza method)")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--engine", required=True, help="kishi-to-classic executable")
    parser.add_argument("--weights", default="random-shogi.weights",
                        help="Weights file path")
    parser.add_argument("--min-rate", type=int, default=1500,
                        help="Minimum player rating (default: 1500)")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games to process (0=all)")
    parser.add_argument("--skip-opening", type=int, default=10,
                        help="Skip first N moves (default: 10)")
    parser.add_argument("--sample-rate", type=float, default=0.5,
                        help="Position sampling rate (default: 0.5)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs (default: 1)")
    args = parser.parse_args()

    kifu_dir = Path(args.kifu)
    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    if args.max_games > 0:
        csa_files = csa_files[:args.max_games]

    # Generate training data
    print("Generating training data...", file=sys.stderr)
    all_samples = []
    games_ok = 0
    skipped = 0

    for filepath in tqdm(csa_files, desc="Parsing kifu", unit="file",
                         file=sys.stderr):
        samples = parse_game_for_classic(
            filepath, min_rate=args.min_rate,
            skip_opening=args.skip_opening,
            sample_rate=args.sample_rate,
        )
        if samples is None:
            skipped += 1
            continue
        all_samples.extend(samples)
        games_ok += 1

    print(f"Parsed {games_ok} games, {len(all_samples)} training positions "
          f"(skipped {skipped})", file=sys.stderr)

    if not all_samples:
        print("No training data. Check kifu path and filters.", file=sys.stderr)
        return 1

    random.shuffle(all_samples)

    # Write training file
    training_file = Path(args.weights).parent / "classic_training.tsv"
    training_file.parent.mkdir(parents=True, exist_ok=True)
    with open(training_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(f"{s['sfen']}\t{s['usi_move']}\n")
    print(f"Wrote {len(all_samples)} samples to {training_file}", file=sys.stderr)

    # Run C++ engine in learn mode
    engine_path = str(Path(args.engine).resolve())
    cmd = [
        engine_path,
        "--learn", str(training_file.resolve()),
        "--weights", str(Path(args.weights).resolve()),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, stderr=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
