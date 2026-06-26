#!/usr/bin/env python3
"""Generate training data for NNUE via engine self-play.

Plays games with a teacher engine (MCTS or NNUE) against itself, recording
search evaluations as training labels. These evaluations provide much higher
quality labels than game outcomes alone.

Output is NPZ format compatible with train_nnue.py --data.

Usage:
    # MCTS as teacher (recommended -- stronger evaluations)
    python tools/nnue_self_play.py --engine build/Release/kishi-to.exe --mcts --games 2000 --movetime 500

    # NNUE self-play
    python tools/nnue_self_play.py --engine build/Release/kishi-to-nnue.exe --games 5000 --movetime 200

    # Parallel with multiple engine instances
    python tools/nnue_self_play.py --engine build/Release/kishi-to.exe --mcts --games 5000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csa_parser import Board, idx, HAND_INDEX, PIECE_TYPE_FROM_ID, parse_csa_initial_board
from train_nnue import extract_features_from_board

BLACK_FIRST_MOVES = [
    "7g7f", "2g2f", "5g5f", "6g6f", "3g3f",
    "4g4f", "5i6h", "7i6h", "5i4h", "3i4h",
]
WHITE_FIRST_MOVES = [
    "3c3d", "8c8d", "5c5d", "4c4d", "3a4b",
    "5a6b", "5a4b", "7a6b", "7c7d", "1c1d",
]

_HIRATE_LINES = [
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


def make_startpos() -> Board:
    board = parse_csa_initial_board(_HIRATE_LINES)
    board.side = 1
    return board


def parse_usi_move(usi_str: str):
    """Convert USI move to (from_sq, to_sq, is_drop, drop_piece, promote)."""
    if len(usi_str) >= 4 and usi_str[1] == '*':
        piece_map = {'P': 1, 'L': 2, 'N': 3, 'S': 4, 'G': 5, 'B': 6, 'R': 7}
        pt = piece_map.get(usi_str[0], 0)
        to_file = int(usi_str[2])
        to_rank = ord(usi_str[3]) - ord('a') + 1
        return -1, idx(to_file, to_rank), True, pt, False
    from_file = int(usi_str[0])
    from_rank = ord(usi_str[1]) - ord('a') + 1
    to_file = int(usi_str[2])
    to_rank = ord(usi_str[3]) - ord('a') + 1
    promote = len(usi_str) >= 5 and usi_str[4] == '+'
    return idx(from_file, from_rank), idx(to_file, to_rank), False, 0, promote


def apply_usi_move(board: Board, usi_str: str):
    """Apply a USI move to the board and flip the side."""
    from_sq, to_sq, is_drop, drop_pt, promote = parse_usi_move(usi_str)
    side = board.side
    board.apply_move(from_sq, to_sq, is_drop, drop_pt, promote, side)
    board.side = -side


class USIEngine:
    """Communicate with a USI engine via subprocess."""

    def __init__(self, exe_path: str, hash_mb: int = 64, extra_options: dict = None):
        self.proc = subprocess.Popen(
            [str(exe_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._send("usi")
        self._wait_for("usiok")
        self._send("setoption name Book value false")
        self._send(f"setoption name Hash value {hash_mb}")
        if extra_options:
            for k, v in extra_options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok", timeout=60)

    def _read_loop(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            with self._lock:
                self._lines.append(line.strip())

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix: str, timeout: int = 30) -> list[str]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for i, line in enumerate(self._lines):
                    if line.startswith(prefix):
                        result = list(self._lines[: i + 1])
                        self._lines = self._lines[i + 1 :]
                        return result
            time.sleep(0.01)
        with self._lock:
            result = list(self._lines)
            self._lines = []
        return result

    def search(self, position_cmd: str, movetime: int) -> tuple[str | None, int | None]:
        """Run a search and return (bestmove, score_cp)."""
        with self._lock:
            self._lines.clear()
        self._send(position_cmd)
        self._send(f"go movetime {movetime}")
        lines = self._wait_for("bestmove", timeout=movetime // 1000 + 15)

        cp = None
        bestmove = None
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) >= 2 else None
            elif line.startswith("info") and "score" in line:
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    cp = int(m.group(1))
                m = re.search(r"score mate (-?\d+)", line)
                if m:
                    cp = 30000 if int(m.group(1)) > 0 else -30000
        return bestmove, cp

    def new_game(self):
        self._send("usinewgame")

    def quit(self):
        self._send("quit")
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def cp_to_label(cp: int, is_mcts: bool) -> float:
    """Convert score_cp to a soft label in [0, 1] from side-to-move perspective."""
    if is_mcts:
        return max(0.0, min(1.0, cp / 1000.0))
    else:
        return 1.0 / (1.0 + np.exp(-cp / 600.0))


def play_game(
    engine: USIEngine,
    movetime: int,
    is_mcts: bool,
    random_plies: int = 4,
    max_plies: int = 512,
) -> list[dict]:
    """Play one self-play game, return training samples."""
    engine.new_game()
    board = make_startpos()
    moves: list[str] = []
    samples: list[dict] = []

    # Pick random opening moves
    opening_moves = []
    if random_plies >= 1:
        opening_moves.append(random.choice(BLACK_FIRST_MOVES))
    if random_plies >= 2:
        opening_moves.append(random.choice(WHITE_FIRST_MOVES))

    for ply in range(max_plies):
        pos_cmd = "position startpos"
        if moves:
            pos_cmd += " moves " + " ".join(moves)

        if ply < len(opening_moves):
            bestmove = opening_moves[ply]
            cp = None
        elif ply < random_plies:
            bestmove, cp = engine.search(pos_cmd, 50)
        else:
            bestmove, cp = engine.search(pos_cmd, movetime)

        if bestmove is None or bestmove in ("resign", "win"):
            break

        if ply >= random_plies and cp is not None:
            bf, wf = extract_features_from_board(board)
            if bf and wf:
                samples.append({
                    "black_feats": np.array(bf, dtype=np.int32),
                    "white_feats": np.array(wf, dtype=np.int32),
                    "side_black": board.side == 1,
                    "soft_label": float(cp_to_label(cp, is_mcts)),
                })

        try:
            apply_usi_move(board, bestmove)
        except Exception:
            break
        moves.append(bestmove)

    return samples


def save_data(samples: list[dict], output_path: str, meta: dict):
    """Save training samples to NPZ compatible with train_nnue.py --data."""
    n = len(samples)
    if n == 0:
        print("No samples to save!")
        return

    all_black = np.concatenate([s["black_feats"] for s in samples])
    all_white = np.concatenate([s["white_feats"] for s in samples])
    black_lengths = np.array([len(s["black_feats"]) for s in samples], dtype=np.int32)
    white_lengths = np.array([len(s["white_feats"]) for s in samples], dtype=np.int32)
    side_black = np.array([s["side_black"] for s in samples], dtype=np.bool_)
    soft_label = np.array([s["soft_label"] for s in samples], dtype=np.float32)

    np.savez(
        output_path,
        black_feats=all_black,
        white_feats=all_white,
        black_lengths=black_lengths,
        white_lengths=white_lengths,
        side_black=side_black,
        soft_label=soft_label,
        fingerprint=np.array([f"selfplay_{n}"]),
        meta_json=np.array([json.dumps(meta)]),
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {n:,} positions to {output_path} ({size_mb:.0f} MB)")


def worker_fn(
    worker_id: int,
    game_ids: list[int],
    exe_path: str,
    movetime: int,
    is_mcts: bool,
    random_plies: int,
    hash_mb: int,
    extra_options: dict | None,
) -> list[dict]:
    """Worker: run multiple games with a dedicated engine instance."""
    engine = USIEngine(exe_path, hash_mb, extra_options)
    samples: list[dict] = []
    t0 = time.time()

    for i, gid in enumerate(game_ids):
        game_samples = play_game(engine, movetime, is_mcts, random_plies)
        samples.extend(game_samples)
        if (i + 1) % 10 == 0 or i == len(game_ids) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(game_ids) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [Worker {worker_id}] Game {i+1}/{len(game_ids)}, "
                f"{len(samples):,} pos, {rate:.1f} games/min, "
                f"ETA {remaining/60:.0f}min"
            )

    engine.quit()
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate NNUE training data via engine self-play"
    )
    parser.add_argument(
        "--engine", required=True, help="Path to engine executable"
    )
    parser.add_argument(
        "--mcts",
        action="store_true",
        help="Engine is MCTS (eval = winRate * 1000). Default assumes NNUE (eval = logit * 600)",
    )
    parser.add_argument(
        "--games", type=int, default=1000, help="Number of games (default: 1000)"
    )
    parser.add_argument(
        "--movetime", type=int, default=200, help="Milliseconds per move (default: 200)"
    )
    parser.add_argument(
        "--random-plies",
        type=int,
        default=4,
        help="Random opening plies for diversity (default: 4)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel engine instances (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="selfplay_data.npz",
        help="Output NPZ file (default: selfplay_data.npz)",
    )
    parser.add_argument(
        "--hash", type=int, default=64, help="Hash table MB per engine (default: 64)"
    )
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="Extra USI option: --engine-option 'Name=Value'",
    )
    args = parser.parse_args()

    extra_options = {}
    for opt in args.engine_option:
        if "=" in opt:
            k, v = opt.split("=", 1)
            extra_options[k.strip()] = v.strip()

    teacher = "MCTS" if args.mcts else "NNUE"
    print(f"Self-play data generation")
    print(f"  Teacher: {teacher} ({args.engine})")
    print(f"  Games: {args.games}, Movetime: {args.movetime}ms")
    print(f"  Workers: {args.workers}, Hash: {args.hash}MB")
    print(f"  Random opening plies: {args.random_plies}")
    print(f"  Output: {args.output}")

    est_time = args.games * 120 * (args.movetime / 1000) / args.workers
    print(f"  Estimated time: {est_time/3600:.1f} hours")
    print()

    game_ids = list(range(args.games))
    all_samples: list[dict] = []
    t0 = time.time()

    if args.workers == 1:
        all_samples = worker_fn(
            0, game_ids, args.engine, args.movetime, args.mcts,
            args.random_plies, args.hash, extra_options,
        )
    else:
        chunk_size = (args.games + args.workers - 1) // args.workers
        chunks = [
            game_ids[i : i + chunk_size]
            for i in range(0, args.games, chunk_size)
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    worker_fn, i, chunk, args.engine, args.movetime,
                    args.mcts, args.random_plies, args.hash, extra_options,
                ): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    worker_samples = future.result()
                    all_samples.extend(worker_samples)
                    print(
                        f"  Worker {wid} finished: {len(worker_samples):,} positions"
                    )
                except Exception as e:
                    print(f"  Worker {wid} failed: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_samples):,} positions from {args.games} games "
          f"in {elapsed/60:.1f} minutes")

    if all_samples:
        labels = np.array([s["soft_label"] for s in all_samples])
        print(f"  Label stats: mean={labels.mean():.3f}, std={labels.std():.3f}, "
              f"min={labels.min():.3f}, max={labels.max():.3f}")

        meta = {
            "source": "self_play",
            "teacher": teacher,
            "engine": args.engine,
            "movetime": args.movetime,
            "games": args.games,
            "random_plies": args.random_plies,
        }
        save_data(all_samples, args.output, meta)


if __name__ == "__main__":
    main()
