#!/usr/bin/env python3
"""Self-play for Alpha engine with training data generation.

Generates .npz files with positions, MCTS visit distributions, and game outcomes
for reinforcement learning.

Usage:
  python alpha_self_play.py --engine build/Release/kishi-to-alpha.exe --games 200 --output selfplay_data.npz
  python alpha_self_play.py --engine build/Release/kishi-to-alpha.exe --games 50 --simulations 400
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional


def _ensure_cuda_lib_path():
    """Add pip-installed NVIDIA CUDA 12 libraries to LD_LIBRARY_PATH."""
    dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.cudnn"):
        try:
            mod = __import__(pkg, fromlist=["lib"])
            if hasattr(mod, "__file__") and mod.__file__:
                lib_dir = str(Path(mod.__file__).parent / "lib")
            elif hasattr(mod, "__path__"):
                lib_dir = str(Path(list(mod.__path__)[0]) / "lib")
            else:
                continue
            if Path(lib_dir).is_dir():
                dirs.append(lib_dir)
        except ImportError:
            pass
    if dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        missing = [d for d in dirs if d not in existing]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(missing) + ((":" + existing) if existing else "")

_ensure_cuda_lib_path()

from csa_parser import (
    Board, idx, file_of, rank_of, CSA_PIECE_MAP, PIECE_TYPE_FROM_ID, HAND_INDEX,
    compute_attacks, move_to_index,
)
from tqdm import tqdm


MAX_MOVES = 256
MOVE_TIMEOUT = 60
POLICY_SIZE = 2187


class AlphaUSIEngine:
    def __init__(self, engine_path: str, model: Optional[str] = None,
                 device: Optional[str] = None, simulations: Optional[int] = None,
                 batch_size: Optional[int] = None, temperature_drop: int = 30):
        self.engine_path = engine_path
        self.model = model
        self.device = device
        self.simulations = simulations
        self.batch_size = batch_size
        self.temperature_drop = temperature_drop
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        engine = str(Path(self.engine_path).resolve())
        self.proc = subprocess.Popen(
            [engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("usi")
        while True:
            line = self._recv()
            if line is None:
                raise RuntimeError("Engine died during USI init")
            if line.strip() == "usiok":
                break

        if self.model:
            self._send(f"setoption name NNModel value {self.model}")
        if self.device:
            self._send(f"setoption name NNDevice value {self.device}")
        if self.simulations is not None:
            self._send(f"setoption name MctsSimulations value {self.simulations}")
        if self.batch_size is not None:
            self._send(f"setoption name MctsBatchSize value {self.batch_size}")
        self._send(f"setoption name TemperatureDropMove value {self.temperature_drop}")
        self._send("setoption name Book value false")

        self._send("isready")
        while True:
            line = self._recv()
            if line is None:
                raise RuntimeError("Engine died during isready")
            if line.strip() == "readyok":
                break

    def new_game(self):
        self._send("usinewgame")

    def go(self, position_cmd: str, movetime: int) -> tuple[str, dict[str, int]]:
        """Returns (bestmove, visit_distribution).

        visit_distribution maps move_index -> visit_count.
        """
        self._send(position_cmd)
        self._send(f"go movetime {movetime}")

        bestmove = "resign"
        deadline = time.monotonic() + MOVE_TIMEOUT + movetime / 1000.0
        while True:
            if time.monotonic() > deadline:
                return "resign", {}
            line = self._recv()
            if line is None:
                return "resign", {}
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else "resign"
                break

        visits = {}
        if bestmove != "resign":
            self._send("getvisits")
            visit_deadline = time.monotonic() + 5.0
            while True:
                if time.monotonic() > visit_deadline:
                    break
                line = self._recv()
                if line is None:
                    break
                if line.startswith("visits"):
                    parts = line.split()[1:]
                    for part in parts:
                        tokens = part.split(":")
                        if len(tokens) >= 3:
                            move_idx = int(tokens[2])
                            count = int(tokens[1])
                            if move_idx >= 0:
                                visits[move_idx] = count
                    break

        return bestmove, visits

    def gameover(self, result: str):
        self._send(f"gameover {result}")

    def quit(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def _recv(self) -> Optional[str]:
        if not self.proc or not self.proc.stdout:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        return line.rstrip("\n\r")


def detect_repetition(moves: list[str]) -> bool:
    if len(moves) < 16:
        return False
    for cycle_len in (4, 6, 8):
        need = cycle_len * 4
        if len(moves) < need:
            continue
        tail = moves[-need:]
        chunks = [tail[i * cycle_len:(i + 1) * cycle_len] for i in range(4)]
        if chunks[0] == chunks[1] == chunks[2] == chunks[3]:
            return True
    return False


def usi_to_board_move(board: Board, usi_move: str):
    """Parse USI move and apply to board. Returns (from_sq, to_sq, is_drop, drop_piece, promote)."""
    if len(usi_move) < 4:
        return None

    if usi_move[1] == '*':
        drop_map = {'P': 1, 'L': 2, 'N': 3, 'S': 4, 'G': 5, 'B': 6, 'R': 7}
        drop_piece = drop_map.get(usi_move[0], 0)
        if drop_piece == 0:
            return None
        to_file = int(usi_move[2])
        to_rank = ord(usi_move[3]) - ord('a') + 1
        to_sq = idx(to_file, to_rank)
        board.apply_move(-1, to_sq, True, drop_piece, False, board.side)
        return (-1, to_sq, True, drop_piece, False)
    else:
        from_file = int(usi_move[0])
        from_rank = ord(usi_move[1]) - ord('a') + 1
        to_file = int(usi_move[2])
        to_rank = ord(usi_move[3]) - ord('a') + 1
        promote = len(usi_move) > 4 and usi_move[4] == '+'
        from_sq = idx(from_file, from_rank)
        to_sq = idx(to_file, to_rank)

        moving_piece = abs(board.squares[from_sq])
        if promote:
            if 1 <= moving_piece <= 4:
                dest_piece = moving_piece + 8
            elif moving_piece == 6:
                dest_piece = 13
            elif moving_piece == 7:
                dest_piece = 14
            else:
                dest_piece = moving_piece
        else:
            dest_piece = moving_piece

        board.apply_move(from_sq, to_sq, False, dest_piece, promote, board.side)
        return (from_sq, to_sq, False, 0, promote)


def hirate_board() -> Board:
    """Create standard starting position board."""
    from csa_parser import parse_csa_initial_board
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
    return parse_csa_initial_board(std_lines)


def play_one_game(engine: AlphaUSIEngine, movetime: int, game_idx: int):
    """Play one self-play game, collecting training data.

    Returns list of (encoded, visit_distribution, side) tuples and game result.
    """
    engine.new_game()
    board = hirate_board()
    moves: list[str] = []
    positions = []

    pbar = tqdm(total=MAX_MOVES, desc=f"Game {game_idx}", unit="mv",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}]",
                file=sys.stderr, leave=False, dynamic_ncols=True, miniters=1)

    result = "draw"
    try:
        for ply in range(MAX_MOVES):
            if moves:
                pos_cmd = "position startpos moves " + " ".join(moves)
            else:
                pos_cmd = "position startpos"

            encoded = board.encode()
            side = board.side

            bestmove, visits = engine.go(pos_cmd, movetime)
            pbar.update(1)

            if bestmove == "resign" or bestmove == "none":
                result = "white" if ply % 2 == 0 else "black"
                break

            if visits:
                positions.append({
                    "encoded": encoded,
                    "visits": visits,
                    "side": side,
                    "ply": ply,
                })

            usi_to_board_move(board, bestmove)
            moves.append(bestmove)

            if detect_repetition(moves):
                result = "draw"
                break
    finally:
        pbar.close()

    engine.gameover("draw" if result == "draw" else
                    ("win" if result == "black" else "lose"))

    return positions, result, len(moves)


def positions_to_arrays(all_positions, all_results):
    """Convert collected positions to numpy arrays for training."""
    total = sum(len(p) for p in all_positions)
    if total == 0:
        return None

    encoded_arr = np.zeros((total, 258), dtype=np.int16)
    policy_arr = np.zeros((total, POLICY_SIZE), dtype=np.float32)
    wdl_arr = np.zeros((total, 3), dtype=np.float32)

    idx = 0
    for positions, result in zip(all_positions, all_results):
        for pos in positions:
            enc = pos["encoded"]
            encoded_arr[idx, :len(enc)] = enc

            visits = pos["visits"]
            total_visits = sum(visits.values())
            if total_visits > 0:
                for move_idx, count in visits.items():
                    if 0 <= move_idx < POLICY_SIZE:
                        policy_arr[idx, move_idx] = count / total_visits

            side = pos["side"]
            if result == "draw":
                wdl_arr[idx] = [0.0, 1.0, 0.0]
            elif (result == "black" and side == 1) or (result == "white" and side == -1):
                wdl_arr[idx] = [1.0, 0.0, 0.0]
            else:
                wdl_arr[idx] = [0.0, 0.0, 1.0]

            idx += 1

    return encoded_arr[:idx], policy_arr[:idx], wdl_arr[:idx]


def main():
    parser = argparse.ArgumentParser(description="Alpha engine self-play with training data")
    parser.add_argument("--engine", required=True, help="Path to kishi-to-alpha executable")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--device", default=None, help="Device (auto/cuda/cpu)")
    parser.add_argument("--games", type=int, default=200, help="Number of games")
    parser.add_argument("--movetime", type=int, default=1000, help="Time per move (ms)")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature-drop", type=int, default=30)
    parser.add_argument("--output", default="selfplay_data.npz", help="Output .npz file")
    args = parser.parse_args()

    engine = AlphaUSIEngine(
        args.engine, model=args.model, device=args.device,
        simulations=args.simulations, batch_size=args.batch_size,
        temperature_drop=args.temperature_drop,
    )
    engine.start()

    all_positions = []
    all_results = []
    wins_black = 0
    wins_white = 0
    draws = 0

    try:
        for game_idx in range(args.games):
            positions, result, ply_count = play_one_game(
                engine, args.movetime, game_idx + 1)

            all_positions.append(positions)
            all_results.append(result)

            if result == "black":
                wins_black += 1
            elif result == "white":
                wins_white += 1
            else:
                draws += 1

            total = wins_black + wins_white + draws
            tqdm.write(
                f"  Game {game_idx+1}/{args.games}: {result} ({ply_count} moves) "
                f"[B:{wins_black} W:{wins_white} D:{draws} "
                f"pos:{sum(len(p) for p in all_positions)}]",
                file=sys.stderr)
    finally:
        engine.quit()

    arrays = positions_to_arrays(all_positions, all_results)
    if arrays is None:
        print("No positions collected.", file=sys.stderr)
        return 1

    encoded, policy, wdl = arrays
    np.savez(args.output, encoded=encoded, policy_target=policy, wdl_target=wdl)
    print(f"Saved {len(encoded)} positions to {args.output}", file=sys.stderr)
    print(f"B:{wins_black} W:{wins_white} D:{draws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
