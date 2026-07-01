#!/usr/bin/env python3
"""Multi-engine arena: tournament-style training loop for all engine types.

Supports Classic, MCTS, NNUE, and Alpha engines. Tracks Elo ratings,
performs smart matchmaking (weak vs strong when gap is large, self-play
when balanced), collects engine-specific training data, and triggers
training + gating after each round.

Usage:
  python tools/arena.py \
    --classic-engine build/kishi-to-classic \
    --nnue-engine build/kishi-to-nnue --nnue-weights best.bin \
    --games-per-round 20 --iterations 50 --movetime 500

  python tools/arena.py \
    --alpha-engine build/kishi-to-alpha --alpha-model best.onnx \
    --nnue-engine build/kishi-to-nnue --nnue-weights best.bin \
    --mcts-engine build/kishi-to --mcts-model nn_model.onnx \
    --classic-engine build/kishi-to-classic \
    --work-dir arena_work --iterations 50
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

def _ensure_cuda_lib_path():
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csa_parser import Board, idx, HAND_INDEX, PIECE_TYPE_FROM_ID, parse_csa_initial_board
from alpha_self_play import (
    detect_repetition,
    hirate_board,
    positions_to_arrays,
    usi_to_board_move,
    POLICY_SIZE,
)
from nnue_self_play import (
    apply_usi_move,
    cp_to_label,
    save_data,
    make_startpos,
)
from train_nnue import extract_features_from_board


MAX_MOVES = 256
MOVE_TIMEOUT = 60


def detect_hardware() -> tuple[int, int]:
    """Auto-detect CPU cores and GPU count.

    Returns (cpu_cores, gpu_count).
    """
    cpu_cores = os.cpu_count() or 4
    gpu_count = 0
    gpu_names: list[str] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_names = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            gpu_count = len(gpu_names)
    except Exception:
        pass
    if gpu_count > 0:
        print(f"  Hardware: {cpu_cores} CPU cores, {gpu_count} GPUs "
              f"({', '.join(gpu_names)})", file=sys.stderr)
    else:
        print(f"  Hardware: {cpu_cores} CPU cores, no GPU detected",
              file=sys.stderr)
    return cpu_cores, gpu_count


# ---------------------------------------------------------------------------
# Unified USI engine wrapper
# ---------------------------------------------------------------------------

class ArenaEngine:
    """USI engine wrapper with engine-type-aware data collection."""

    def __init__(self, name: str, exe_path: str, engine_type: str,
                 options: dict | None = None):
        self.name = name
        self.exe_path = exe_path
        self.engine_type = engine_type  # "classic", "mcts", "nnue", "alpha"
        self.options = options or {}
        self.proc: Optional[subprocess.Popen] = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None

    def start(self):
        engine = str(Path(self.exe_path).resolve())
        self.proc = subprocess.Popen(
            [engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lines = []
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._send("usi")
        self._wait_for("usiok")

        for k, v in self.options.items():
            self._send(f"setoption name {k} value {v}")

        self._send("isready")
        lines = self._wait_for("readyok", timeout=60)
        errors = [l for l in lines
                  if "ERROR" in l or "random weights" in l]
        if errors:
            for e in errors:
                print(f"  [{self.name}] {e}", file=sys.stderr)
            raise RuntimeError(f"Engine '{self.name}' failed to initialize: {errors[0]}")

    def restart(self):
        self.quit()
        self.start()

    def new_game(self):
        self._send("usinewgame")

    def go(self, pos_cmd: str, movetime: int) -> tuple[str, int | None, dict[str, int] | None]:
        """Returns (bestmove, score_cp, visits_dict).

        visits_dict is only populated for Alpha engines.
        """
        with self._lock:
            self._lines.clear()
        self._send(pos_cmd)
        self._send(f"go movetime {movetime}")

        lines = self._wait_for("bestmove", timeout=movetime // 1000 + MOVE_TIMEOUT)

        bestmove = "resign"
        cp = None
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) >= 2 else "resign"
            elif line.startswith("info") and "score" in line:
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    cp = int(m.group(1))
                m = re.search(r"score mate (-?\d+)", line)
                if m:
                    cp = 30000 if int(m.group(1)) > 0 else -30000

        visits = None
        if self.engine_type == "alpha" and bestmove not in ("resign", "none"):
            visits = self._get_visits()

        return bestmove, cp, visits

    def _get_visits(self) -> dict[int, int]:
        with self._lock:
            self._lines.clear()
        self._send("getvisits")
        lines = self._wait_for("visits", timeout=5)
        visits: dict[int, int] = {}
        for line in lines:
            if line.startswith("visits"):
                parts = line.split()[1:]
                for part in parts:
                    tokens = part.split(":")
                    if len(tokens) >= 3:
                        move_idx = int(tokens[2])
                        count = int(tokens[1])
                        if move_idx >= 0:
                            visits[move_idx] = count
        return visits

    def gameover(self, result: str):
        self._send(f"gameover {result}")

    def quit(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def _read_loop(self):
        while True:
            if not self.proc or not self.proc.stdout:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            with self._lock:
                self._lines.append(line.strip())

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin and self.proc.poll() is None:
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    def _wait_for(self, prefix: str, timeout: int = 30) -> list[str]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for i, line in enumerate(self._lines):
                    if line.startswith(prefix):
                        result = list(self._lines[:i + 1])
                        self._lines = self._lines[i + 1:]
                        return result
            time.sleep(0.01)
        with self._lock:
            result = list(self._lines)
            self._lines = []
        return result


# ---------------------------------------------------------------------------
# Elo tracker
# ---------------------------------------------------------------------------

class EloTracker:
    K = 32
    INITIAL = 1500

    def __init__(self, names: list[str]):
        self.ratings: dict[str, float] = {n: self.INITIAL for n in names}
        self.history: list[dict] = []

    def update(self, a: str, b: str, result: float):
        """result: 1.0 = a wins, 0.0 = b wins, 0.5 = draw."""
        ra, rb = self.ratings[a], self.ratings[b]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        self.ratings[a] += self.K * (result - ea)
        self.ratings[b] += self.K * ((1.0 - result) - eb)

    def gap(self) -> float:
        if len(self.ratings) < 2:
            return 0.0
        vals = list(self.ratings.values())
        return max(vals) - min(vals)

    def weakest(self) -> str:
        return min(self.ratings, key=self.ratings.get)

    def strongest(self) -> str:
        return max(self.ratings, key=self.ratings.get)

    def snapshot(self) -> dict:
        return dict(self.ratings)

    def save(self, path: str):
        data = {"ratings": self.ratings, "history": self.history}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not Path(path).exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.ratings.update(data.get("ratings", {}))
        self.history = data.get("history", [])


# ---------------------------------------------------------------------------
# Game play & data collection
# ---------------------------------------------------------------------------

RESIGN_THRESHOLD = -2000

def play_arena_game(
    black: ArenaEngine,
    white: ArenaEngine,
    movetime: int,
    on_move: Optional[callable] = None,
    opening_moves: list[str] | None = None,
) -> dict:
    """Play one game, collecting engine-type-specific training data.

    Returns a game record dict with moves, result, and per-engine data.
    """
    black.new_game()
    white.new_game()

    board_csa = hirate_board()
    board_nnue = make_startpos()
    moves: list[str] = list(opening_moves) if opening_moves else []
    start_ply = len(moves)
    for m in (opening_moves or []):
        usi_to_board_move(board_csa, m)
        apply_usi_move(board_nnue, m)
    result = "draw"

    alpha_positions: list[dict] = []
    nnue_samples: list[dict] = []
    mcts_positions: list[dict] = []

    imitation_plies: list[dict] = []

    for ply in range(start_ply, MAX_MOVES):
        current = black if ply % 2 == 0 else white

        pos_cmd = "position startpos"
        if moves:
            pos_cmd += " moves " + " ".join(moves)

        encoded = board_csa.encode()
        side = board_csa.side
        bf, wf = extract_features_from_board(board_nnue)
        side_black = board_nnue.side == 1

        try:
            bestmove, cp, visits = current.go(pos_cmd, movetime)
        except (OSError, BrokenPipeError):
            bestmove = "resign"
            cp = None
            visits = None

        if bestmove in ("resign", "none", None) or (cp is not None and cp <= RESIGN_THRESHOLD):
            if cp is not None and cp <= RESIGN_THRESHOLD and bestmove not in ("resign", "none", None):
                bestmove = "resign"
            if ply % 2 == 0:
                result = "white"
            else:
                result = "black"
            if on_move:
                winner = "white" if ply % 2 == 0 else "black"
                on_move(ply + 1, bestmove, cp, winner)
            break

        if on_move:
            on_move(ply + 1, bestmove, cp, None)

        imitation_plies.append({
            "ply": ply,
            "encoded": encoded,
            "sfen": board_csa.to_sfen(move_number=ply + 1),
            "side": side,
            "bf": bf,
            "wf": wf,
            "side_black": side_black,
            "bestmove": bestmove,
            "cp": cp,
            "mover": current.name,
        })

        if current.engine_type == "alpha" and visits:
            alpha_positions.append({
                "encoded": encoded,
                "visits": visits,
                "side": side,
                "ply": ply,
            })
        elif current.engine_type == "nnue" and cp is not None:
            if bf and wf:
                nnue_samples.append({
                    "black_feats": np.array(bf, dtype=np.int32),
                    "white_feats": np.array(wf, dtype=np.int32),
                    "side_black": side_black,
                    "soft_label": float(cp_to_label(cp, False)),
                    "ply": ply,
                })
        elif current.engine_type == "mcts" and cp is not None:
            from csa_parser import move_to_index
            from nnue_self_play import parse_usi_move
            parsed = parse_usi_move(bestmove)
            if parsed:
                from_sq, to_sq, is_drop, drop_pt, promote = parsed
                midx = move_to_index(from_sq, to_sq, is_drop, drop_pt, promote, side)
                if midx >= 0:
                    mcts_positions.append({
                        "encoded": encoded,
                        "move_index": midx,
                        "value": cp / 1000.0,
                        "ply": ply,
                    })

        usi_to_board_move(board_csa, bestmove)
        apply_usi_move(board_nnue, bestmove)
        moves.append(bestmove)

        if detect_repetition(moves):
            result = "draw"
            break

    black_result = "draw"
    white_result = "draw"
    if result == "black":
        black_result = "win"
        white_result = "lose"
    elif result == "white":
        black_result = "lose"
        white_result = "win"
    black.gameover(black_result)
    white.gameover(white_result)

    _blend_labels_with_outcome(nnue_samples, result, len(moves))

    return {
        "black": black.name,
        "white": white.name,
        "result": result,
        "ply_count": len(moves),
        "moves": moves,
        "alpha_positions": alpha_positions,
        "nnue_samples": nnue_samples,
        "mcts_positions": mcts_positions,
        "imitation_plies": imitation_plies,
    }


def _blend_labels_with_outcome(
    nnue_samples: list[dict],
    result: str,
    total_plies: int,
    eval_weight: float = 0.7,
    halflife: float = 60.0,
):
    for sample in nnue_samples:
        is_black = sample["side_black"]
        if result == "draw":
            outcome = 0.5
        elif (result == "black" and is_black) or (result == "white" and not is_black):
            outcome = 1.0
        else:
            outcome = 0.0
        plies_remaining = total_plies - sample["ply"]
        discount = 0.5 ** (plies_remaining / halflife)
        outcome_label = 0.5 + discount * (outcome - 0.5)
        sample["soft_label"] = eval_weight * sample["soft_label"] + (1 - eval_weight) * outcome_label


def _imitation_for_loser(
    imitation_plies: list[dict],
    winner_name: str,
    loser_type: str,
    alpha_positions: list[list[dict]],
    alpha_results: list[str],
    nnue_samples: list[dict],
    mcts_positions: list[dict],
):
    """Convert winner's moves into training data for the loser's engine type."""
    from csa_parser import move_to_index
    from nnue_self_play import parse_usi_move

    winner_plies = [p for p in imitation_plies if p["mover"] == winner_name]
    if not winner_plies:
        return

    if loser_type == "mcts":
        for p in winner_plies:
            parsed = parse_usi_move(p["bestmove"])
            if not parsed:
                continue
            from_sq, to_sq, is_drop, drop_pt, promote = parsed
            midx = move_to_index(from_sq, to_sq, is_drop, drop_pt, promote, p["side"])
            if midx >= 0:
                mcts_positions.append({
                    "encoded": p["encoded"],
                    "move_index": midx,
                    "value": 1.0,
                    "ply": p["ply"],
                })

    elif loser_type == "nnue":
        for p in winner_plies:
            if p["bf"] and p["wf"]:
                nnue_samples.append({
                    "black_feats": np.array(p["bf"], dtype=np.int32),
                    "white_feats": np.array(p["wf"], dtype=np.int32),
                    "side_black": p["side_black"],
                    "soft_label": 0.85,
                    "ply": p["ply"],
                })

    elif loser_type == "alpha":
        positions = []
        for p in winner_plies:
            parsed = parse_usi_move(p["bestmove"])
            if not parsed:
                continue
            from_sq, to_sq, is_drop, drop_pt, promote = parsed
            midx = move_to_index(from_sq, to_sq, is_drop, drop_pt, promote, p["side"])
            if midx >= 0:
                visits = {midx: 1.0}
                positions.append({
                    "encoded": p["encoded"],
                    "visits": visits,
                    "side": p["side"],
                    "ply": p["ply"],
                })
        if positions:
            alpha_positions.append(positions)
            alpha_results.append("black")


# ---------------------------------------------------------------------------
# Matchmaking
# ---------------------------------------------------------------------------

def select_matchup(
    engine_names: list[str],
    elo: EloTracker,
    gap_threshold: float = 200.0,
    pair_counts: dict[tuple[str, str], int] | None = None,
) -> tuple[str, str]:
    if len(engine_names) < 2:
        return engine_names[0], engine_names[0]

    if pair_counts:
        min_count = min(pair_counts.values()) if pair_counts else 0
        least_pairs = [p for p, c in pair_counts.items() if c == min_count]
        a, b = random.choice(least_pairs)
        if random.random() < 0.5:
            a, b = b, a
        return a, b

    if elo.gap() > gap_threshold and random.random() < 0.5:
        weak = elo.weakest()
        others = [n for n in engine_names if n != weak]
        return weak, random.choice(others)
    a, b = random.sample(engine_names, 2)
    return a, b


# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------

def save_alpha_data(positions: list[list[dict]], results: list[str], path: str):
    arrays = positions_to_arrays(positions, results)
    if arrays is None:
        return 0
    encoded, policy, wdl = arrays
    np.savez(path, encoded=encoded, policy_target=policy, wdl_target=wdl)
    return len(encoded)


def save_nnue_data(samples: list[dict], path: str):
    if not samples:
        return 0
    meta = {"source": "arena"}
    save_data(samples, path, meta)
    return len(samples)


def save_mcts_data(positions: list[dict], result_map: dict[int, str], path: str):
    """Save MCTS training data compatible with train.py --cache."""
    if not positions:
        return 0
    n = len(positions)
    encoded_arr = np.zeros((n, 258), dtype=np.int16)
    values_arr = np.zeros(n, dtype=np.float32)
    policies_arr = np.zeros(n, dtype=np.int32)
    for i, pos in enumerate(positions):
        enc = pos["encoded"]
        encoded_arr[i, :len(enc)] = enc
        values_arr[i] = np.clip(pos["value"], -1.0, 1.0)
        policies_arr[i] = pos["move_index"]
    np.savez(path, encoded=encoded_arr, values=values_arr, policies=policies_arr)
    return n


# ---------------------------------------------------------------------------
# Training & gating
# ---------------------------------------------------------------------------

def run_cmd(args: list[str], desc: str) -> int:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {desc}", file=sys.stderr)
    print(f"  cmd: {' '.join(args)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    t0 = time.time()
    result = subprocess.run(args)
    elapsed = time.time() - t0
    print(f"  -> exit={result.returncode} ({elapsed:.0f}s)", file=sys.stderr)
    return result.returncode


def _play_gate_game(engine_new, engine_best, new_is_black: bool,
                    movetime: int, on_move=None,
                    opening_moves: list[str] | None = None) -> tuple[str, int]:
    """Play a single gate game between new and best engines.

    Returns (result, ply_count) where result is 'black', 'white', or 'draw'.
    opening_moves: pre-played moves from opening suite (both engines skip these).
    """
    engine_new.new_game()
    engine_best.new_game()

    b = engine_new if new_is_black else engine_best
    w = engine_best if new_is_black else engine_new
    moves: list[str] = list(opening_moves) if opening_moves else []
    start_ply = len(moves)

    for ply in range(start_ply, MAX_MOVES):
        current = b if ply % 2 == 0 else w
        pos_cmd = "position startpos"
        if moves:
            pos_cmd += " moves " + " ".join(moves)

        try:
            bestmove = current.go(pos_cmd, movetime)
            if isinstance(bestmove, tuple):
                bestmove, cp_val = bestmove[0], bestmove[1] if len(bestmove) > 1 else None
            else:
                cp_val = None
        except (OSError, BrokenPipeError):
            bestmove = "resign"
            cp_val = None

        if bestmove in ("resign", "none", None):
            winner = "white" if ply % 2 == 0 else "black"
            if on_move:
                on_move(ply + 1 - start_ply, cp_val, winner)
            b.gameover("win" if winner == "black" else "lose")
            w.gameover("win" if winner == "white" else "lose")
            return winner, ply + 1 - start_ply

        if on_move:
            on_move(ply + 1 - start_ply, cp_val, None)

        moves.append(bestmove)
        if detect_repetition(moves):
            b.gameover("draw")
            w.gameover("draw")
            return "draw", ply + 1 - start_ply

    b.gameover("draw")
    w.gameover("draw")
    return "draw", MAX_MOVES - start_ply


def evaluate_models(
    engine: ArenaEngine,
    new_model_opts: dict,
    best_model_opts: dict,
    games: int,
    movetime: int,
    gate_threshold: float = 0.55,
    parallel: int = 1,
    opening_suite: list[str] | None = None,
) -> float:
    """Play new vs best model, return win rate for new."""
    from self_play import USIEngine as SPEngine

    n_parallel = max(1, min(parallel, games))

    engine_pairs: list[tuple] = []
    for _ in range(n_parallel):
        e_new = SPEngine(engine.exe_path)
        e_best = SPEngine(engine.exe_path)
        e_new.start()
        e_best.start()
        for k, v in new_model_opts.items():
            e_new._send(f"setoption name {k} value {v}")
        for k, v in best_model_opts.items():
            e_best._send(f"setoption name {k} value {v}")
        e_new._send("isready")
        e_best._send("isready")
        for eng in (e_new, e_best):
            while True:
                line = eng._recv()
                if line and line.strip() == "readyok":
                    break
        engine_pairs.append((e_new, e_best))

    try:
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
        from rich.panel import Panel
        from rich.layout import Layout
        has_rich = True
    except ImportError:
        has_rich = False

    wins = 0
    losses = 0
    draws = 0
    wins_as_black = 0
    wins_as_white = 0
    losses_as_black = 0
    losses_as_white = 0
    lock = threading.Lock()
    early_stop = False

    slot_status: list[dict] = [{"state": "idle"} for _ in range(n_parallel)]
    completed_lines: list[str] = []
    live_ctx = [None]

    def _build_gate_display():
        total = wins + losses + draws
        wr = (wins + draws * 0.5) / total if total > 0 else 0.5
        summary = (f"W={wins}(B{wins_as_black}/W{wins_as_white}) "
                   f"L={losses}(B{losses_as_black}/W{losses_as_white}) "
                   f"D={draws}  WR={wr:.1%}  [{total}/{games}]")

        table = Table(title="Gate Evaluation", show_header=True, expand=True, padding=(0, 1))
        table.add_column("Slot", style="bold", width=4, justify="center")
        table.add_column("Game", width=8)
        table.add_column("Side", width=6)
        table.add_column("Ply", width=5, justify="right")
        table.add_column("Status", width=14)
        for i, ss in enumerate(slot_status):
            if ss["state"] == "idle":
                table.add_row(str(i + 1), "-", "-", "-", Text("idle", style="dim"))
            elif ss["state"] == "done":
                res = ss.get("result", "")
                style = "green" if res == "win" else ("yellow" if res == "draw" else "red")
                table.add_row(str(i + 1), ss.get("game", ""),
                              ss.get("side", ""),
                              str(ss.get("ply", "")),
                              Text(res, style=style))
            else:
                table.add_row(str(i + 1), ss.get("game", ""),
                              ss.get("side", ""),
                              str(ss.get("ply", 0)),
                              Text("playing", style="cyan"))

        recent = "\n".join(completed_lines[-n_parallel * 2:]) if completed_lines else ""
        log_panel = Panel(recent, title="Results", border_style="dim", expand=True) if recent else None

        layout = Layout()
        parts = [
            Layout(Panel(summary, title="Gate Progress", border_style="blue"), size=3),
            Layout(table, size=n_parallel + 4),
        ]
        if log_panel:
            parts.append(Layout(log_panel, size=min(n_parallel * 2 + 3, 8)))
        layout.split_column(*parts)
        return layout

    def _refresh():
        if has_rich and live_ctx[0]:
            live_ctx[0].update(_build_gate_display())

    shuffled_suite: list[str] = []
    if opening_suite:
        shuffled_suite = list(opening_suite)
        random.shuffle(shuffled_suite)

    def _get_opening(g: int) -> list[str] | None:
        if not shuffled_suite:
            return None
        return shuffled_suite[g % len(shuffled_suite)].split()

    def _play_gate(slot: int, g: int):
        e_new, e_best = engine_pairs[slot]
        new_is_black = g % 2 == 0
        opening = _get_opening(g // 2)
        side_str = "new=B" if new_is_black else "new=W"
        slot_status[slot] = {
            "state": "playing", "game": f"{g+1}/{games}",
            "side": side_str, "ply": 0,
        }
        _refresh()

        def on_move(ply, cp, winner):
            if winner:
                new_won = (winner == "black" and new_is_black) or (winner == "white" and not new_is_black)
                res = "win" if new_won else ("draw" if "draw" in winner else "loss")
                slot_status[slot].update({"state": "done", "ply": ply, "result": res})
            else:
                slot_status[slot]["ply"] = ply
            _refresh()

        result, ply = _play_gate_game(e_new, e_best, new_is_black, movetime,
                                       on_move, opening)
        return g, result, new_is_black, ply

    live_manager = None
    if has_rich and n_parallel > 1:
        live_manager = Live(_build_gate_display(), refresh_per_second=4,
                            console=Console(stderr=True))

    try:
        if live_manager:
            live_manager.start()
            live_ctx[0] = live_manager

        free_slots = list(range(n_parallel))
        games_started = 0
        games_done = 0
        pending: dict[concurrent.futures.Future, int] = {}
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel)

        def _submit_gate():
            nonlocal games_started
            if games_started >= games or not free_slots or early_stop:
                return
            slot = free_slots.pop(0)
            fut = executor.submit(_play_gate, slot, games_started)
            pending[fut] = slot
            games_started += 1

        for _ in range(min(n_parallel, games)):
            _submit_gate()

        while games_done < games and pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                slot = pending.pop(future)
                free_slots.append(slot)
                games_done += 1
                g, result, new_is_black, ply = future.result()

                with lock:
                    if "draw" in result:
                        draws += 1
                    elif (result == "black" and new_is_black) or (result == "white" and not new_is_black):
                        wins += 1
                        if new_is_black:
                            wins_as_black += 1
                        else:
                            wins_as_white += 1
                    else:
                        losses += 1
                        if new_is_black:
                            losses_as_black += 1
                        else:
                            losses_as_white += 1

                    total = wins + losses + draws
                    wr = (wins + draws * 0.5) / total if total > 0 else 0.5
                    remaining = games - games_done
                    best_possible = (wins + draws * 0.5 + remaining) / games
                    worst_possible = (wins + draws * 0.5) / games
                    side = "B" if new_is_black else "W"
                    new_won = (result == "black" and new_is_black) or (result == "white" and not new_is_black)
                    res_str = "W" if new_won else ("D" if "draw" in result else "L")
                    line = (f"  game {g+1}: {res_str} ({ply}mv, {side}) "
                            f"WR={wr:.1%} [{games_done}/{games}]")
                    completed_lines.append(line)
                    if not has_rich or not live_manager:
                        print(f"  gate {games_done}/{games}: W={wins}(B{wins_as_black}/W{wins_as_white}) "
                              f"L={losses}(B{losses_as_black}/W{losses_as_white}) D={draws} "
                              f"WR={wr:.1%} [{side}]",
                              file=sys.stderr)
                    _refresh()

                    if remaining > 0 and best_possible < gate_threshold:
                        msg = f"  gate: early reject (best possible {best_possible:.1%} < {gate_threshold:.1%})"
                        completed_lines.append(msg)
                        if not has_rich or not live_manager:
                            print(msg, file=sys.stderr)
                        early_stop = True
                        _refresh()
                    elif remaining > 0 and worst_possible >= gate_threshold:
                        msg = f"  gate: early accept (worst possible {worst_possible:.1%} >= {gate_threshold:.1%})"
                        completed_lines.append(msg)
                        if not has_rich or not live_manager:
                            print(msg, file=sys.stderr)
                        early_stop = True
                        _refresh()

                if not early_stop:
                    _submit_gate()

        executor.shutdown(wait=True)
    finally:
        if live_manager:
            live_ctx[0] = None
            live_manager.stop()
            total = wins + losses + draws
            wr = (wins + draws * 0.5) / total if total > 0 else 0.5
            print(f"  gate: W={wins}(B{wins_as_black}/W{wins_as_white}) "
                  f"L={losses}(B{losses_as_black}/W{losses_as_white}) D={draws} "
                  f"WR={wr:.1%}", file=sys.stderr)
        for e_new, e_best in engine_pairs:
            e_new.quit()
            e_best.quit()

    total = wins + losses + draws
    return (wins + draws * 0.5) / total if total > 0 else 0.5


def collect_distillation_samples(
    game_record: dict,
    teacher_name: str,
    eval_scale: float = 361.0,
) -> list[tuple[str, str, float]]:
    """Extract (sfen, bestmove, label) from a game record for Classic distillation.

    Uses positions where teacher_name was the mover and provided cp.
    Label = sigmoid(cp / eval_scale) = win probability.
    """
    samples = []
    for ply_data in game_record["imitation_plies"]:
        if ply_data["mover"] != teacher_name or ply_data["cp"] is None:
            continue
        sfen = ply_data.get("sfen")
        if not sfen:
            continue
        cp = ply_data["cp"]
        label = 1.0 / (1.0 + math.exp(-cp / eval_scale))
        label = max(0.01, min(0.99, label))
        samples.append((sfen, ply_data["bestmove"], label))
    return samples


def save_distillation_tsv(samples: list[tuple[str, str, float]], path: str) -> int:
    """Save distillation samples as SFEN+move+label TSV for --extract-features."""
    if not samples:
        return 0
    with open(path, "w") as f:
        for sfen, move, label in samples:
            f.write(f"{sfen}\t{move}\t{label:.6f}\n")
    return len(samples)


def train_and_gate_classic(
    work_dir: Path,
    engine: ArenaEngine,
    iteration: int,
    distill_samples: list[tuple[str, str, float]],
    gate_games: int,
    gate_movetime: int,
    gate_threshold: float,
    gate_parallel: int = 1,
    opening_suite: list[str] | None = None,
) -> bool:
    """Train Classic MLP from distillation data and gate."""
    model_dir = work_dir / "classic" / "models"
    data_dir = work_dir / "classic" / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not distill_samples:
        return False

    positions_file = data_dir / f"distill_{iteration:04d}.tsv"
    n = save_distillation_tsv(distill_samples, str(positions_file))
    print(f"  Classic: {n} distillation samples", file=sys.stderr)

    features_file = data_dir / f"features_{iteration:04d}.tsv"
    engine_path = str(Path(engine.exe_path).resolve())
    extract_cmd = [
        engine_path,
        "--extract-features", str(positions_file.resolve()),
        "--output", str(features_file.resolve()),
        "--negatives", "0",
    ]
    if run_cmd(extract_cmd, f"Classic feature extraction (iter {iteration})") != 0:
        return False

    best_pt = model_dir / "best.pt"
    candidate_pt = model_dir / f"candidate_{iteration}.pt"
    candidate_weights = model_dir / f"candidate_{iteration}.weights"

    if best_pt.exists():
        shutil.copy2(best_pt, candidate_pt)

    train_args = [
        sys.executable, "tools/mlp_eval.py", "train",
        "--data", str(features_file.resolve()),
        "--model", str(candidate_pt),
        "--device", "auto",
        "--epochs", "5",
        "--batch-size", "512",
    ]
    if run_cmd(train_args, f"Classic MLP training (iter {iteration})") != 0:
        return False

    export_args = [
        sys.executable, "tools/export_mlp.py",
        "--model", str(candidate_pt),
        "--output", str(candidate_weights),
    ]
    if run_cmd(export_args, f"Classic MLP export (iter {iteration})") != 0:
        return False

    best_weights = model_dir / "best.weights"
    if not best_weights.exists():
        shutil.copy2(candidate_weights, best_weights)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  Classic: first model promoted as best", file=sys.stderr)
        engine.options["MlpWeightsFile"] = str(best_weights.resolve())
        engine.restart()
        return True

    wr = evaluate_models(
        engine,
        {"MlpWeightsFile": str(candidate_weights.resolve())},
        {"MlpWeightsFile": str(best_weights.resolve())},
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )
    print(f"  Classic gate: WR={wr:.1%} (threshold={gate_threshold:.1%})", file=sys.stderr)

    if wr >= gate_threshold:
        shutil.copy2(candidate_weights, best_weights)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  Classic: new model PROMOTED", file=sys.stderr)
        engine.options["MlpWeightsFile"] = str(best_weights.resolve())
        engine.restart()
        return True
    else:
        print(f"  Classic: new model rejected", file=sys.stderr)
        return False


def train_and_gate_alpha(work_dir: Path, engine: ArenaEngine, iteration: int,
                         gate_games: int, gate_movetime: int,
                         gate_threshold: float, gate_parallel: int = 1,
                         opening_suite: list[str] | None = None) -> bool:
    data_dir = work_dir / "alpha" / "data"
    model_dir = work_dir / "alpha" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        return False

    best_pt = model_dir / "best.pt"
    candidate_pt = model_dir / f"candidate_{iteration}.pt"
    candidate_onnx = model_dir / f"candidate_{iteration}.onnx"

    train_args = [
        sys.executable, "tools/train_alpha_rl.py",
        "--data", *[str(f) for f in npz_files],
        "--model", str(candidate_pt),
        "--output", str(candidate_onnx),
    ]
    if best_pt.exists():
        shutil.copy2(best_pt, candidate_pt)

    if run_cmd(train_args, f"Alpha training (iter {iteration})") != 0:
        return False

    best_onnx = model_dir / "best.onnx"
    if not best_onnx.exists():
        shutil.copy2(candidate_onnx, best_onnx)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  Alpha: first model promoted as best", file=sys.stderr)
        engine.options["NNModel"] = str(best_onnx.resolve())
        engine.restart()
        return True

    wr = evaluate_models(
        engine,
        {"NNModel": str(candidate_onnx.resolve())},
        {"NNModel": str(best_onnx.resolve())},
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )
    print(f"  Alpha gate: WR={wr:.1%} (threshold={gate_threshold:.1%})", file=sys.stderr)

    if wr >= gate_threshold:
        shutil.copy2(candidate_onnx, best_onnx)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  Alpha: new model PROMOTED", file=sys.stderr)
        engine.options["NNModel"] = str(best_onnx.resolve())
        engine.restart()
        return True
    else:
        print(f"  Alpha: new model rejected", file=sys.stderr)
        return False


def train_and_gate_nnue(work_dir: Path, engine: ArenaEngine, iteration: int,
                        gate_games: int, gate_movetime: int,
                        gate_threshold: float, gate_parallel: int = 1,
                        opening_suite: list[str] | None = None) -> bool:
    data_dir = work_dir / "nnue" / "data"
    model_dir = work_dir / "nnue" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        return False

    best_pt = model_dir / "best.pt"
    best_bin = model_dir / "best.bin"
    candidate_pt = model_dir / f"candidate_{iteration}.pt"
    candidate_bin = model_dir / f"candidate_{iteration}.bin"

    train_args = [
        sys.executable, "tools/train_nnue.py",
        "--data", str(npz_files[-1]),
        "--output", str(candidate_bin),
        "--model-pt", str(candidate_pt),
    ]
    if best_pt.exists():
        train_args += ["--resume", str(best_pt)]

    if run_cmd(train_args, f"NNUE training (iter {iteration})") != 0:
        return False

    if not best_bin.exists():
        if candidate_bin.exists():
            shutil.copy2(candidate_bin, best_bin)
        if candidate_pt.exists():
            shutil.copy2(candidate_pt, best_pt)
        print(f"  NNUE: first model promoted as best", file=sys.stderr)
        engine.options["NNUEFile"] = str(best_bin.resolve())
        engine.restart()
        return True

    wr = evaluate_models(
        engine,
        {"NNUEFile": str(candidate_bin.resolve())},
        {"NNUEFile": str(best_bin.resolve())},
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )
    print(f"  NNUE gate: WR={wr:.1%} (threshold={gate_threshold:.1%})", file=sys.stderr)

    if wr >= gate_threshold:
        shutil.copy2(candidate_bin, best_bin)
        if candidate_pt.exists():
            shutil.copy2(candidate_pt, best_pt)
        print(f"  NNUE: new model PROMOTED", file=sys.stderr)
        engine.options["NNUEFile"] = str(best_bin.resolve())
        engine.restart()
        return True
    else:
        print(f"  NNUE: new model rejected", file=sys.stderr)
        return False


def train_and_gate_mcts(work_dir: Path, engine: ArenaEngine, iteration: int,
                        gate_games: int, gate_movetime: int,
                        gate_threshold: float, gate_parallel: int = 1,
                        opening_suite: list[str] | None = None) -> bool:
    data_dir = work_dir / "mcts" / "data"
    model_dir = work_dir / "mcts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        return False

    best_pt = model_dir / "best.pt"
    best_onnx = model_dir / "best.onnx"
    candidate_pt = model_dir / f"candidate_{iteration}.pt"
    candidate_onnx = model_dir / f"candidate_{iteration}.onnx"

    train_args = [
        sys.executable, "tools/train.py",
        "--cache", str(npz_files[-1]),
        "--model", str(candidate_pt),
        "--output", str(candidate_onnx),
    ]
    if best_pt.exists():
        shutil.copy2(best_pt, candidate_pt)
        train_args += ["--resume"]

    if run_cmd(train_args, f"MCTS training (iter {iteration})") != 0:
        return False

    if not best_onnx.exists():
        shutil.copy2(candidate_onnx, best_onnx)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  MCTS: first model promoted as best", file=sys.stderr)
        engine.options["NNModel"] = str(best_onnx.resolve())
        engine.restart()
        return True

    wr = evaluate_models(
        engine,
        {"NNModel": str(candidate_onnx.resolve())},
        {"NNModel": str(best_onnx.resolve())},
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )
    print(f"  MCTS gate: WR={wr:.1%} (threshold={gate_threshold:.1%})", file=sys.stderr)

    if wr >= gate_threshold:
        shutil.copy2(candidate_onnx, best_onnx)
        shutil.copy2(candidate_pt, best_pt)
        print(f"  MCTS: new model PROMOTED", file=sys.stderr)
        engine.options["NNModel"] = str(best_onnx.resolve())
        engine.restart()
        return True
    else:
        print(f"  MCTS: new model rejected", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Self-play training for top engines (Feature A)
# ---------------------------------------------------------------------------

def _merge_alpha_npz(paths: list[Path], output: Path):
    """Merge multiple Alpha self-play npz files into one."""
    all_encoded, all_policy, all_wdl = [], [], []
    for p in paths:
        if not p.exists():
            continue
        data = np.load(str(p))
        if "encoded" in data:
            all_encoded.append(data["encoded"])
            all_policy.append(data["policy_target"])
            all_wdl.append(data["wdl_target"])
    if not all_encoded:
        return
    np.savez(
        str(output),
        encoded=np.concatenate(all_encoded),
        policy_target=np.concatenate(all_policy),
        wdl_target=np.concatenate(all_wdl),
    )


def self_play_and_train_alpha(
    work_dir: Path,
    engine: ArenaEngine,
    iteration: int,
    num_games: int,
    gate_games: int,
    gate_movetime: int,
    gate_threshold: float,
    gate_parallel: int = 1,
    opening_suite: list[str] | None = None,
    gpu_count: int = 1,
    temperature_drop: int = 60,
) -> bool:
    """Run Alpha self-play with parallel GPU workers and diversity."""
    data_dir = work_dir / "alpha" / "data"
    model_dir = work_dir / "alpha" / "models"

    best_onnx = model_dir / "best.onnx"
    if not best_onnx.exists():
        print("  Alpha self-play: no best model, skipping", file=sys.stderr)
        return False

    model_str = str(best_onnx.resolve())
    simulations = engine.options.get("MctsSimulations", "400")
    batch_size = engine.options.get("MctsBatchSize", "16")

    workers = max(1, min(num_games, gpu_count))
    games_per_worker = (num_games + workers - 1) // workers

    print(f"  Alpha self-play: {num_games} games, {workers} workers "
          f"(temperature_drop={temperature_drop})", file=sys.stderr)

    data_paths: list[Path] = []
    procs: list[subprocess.Popen] = []

    for w in range(workers):
        n = min(games_per_worker, num_games - w * games_per_worker)
        if n <= 0:
            break
        data_path = data_dir / f"selfplay_{iteration:04d}_w{w}.npz"
        data_paths.append(data_path)

        device = f"cuda:{w}" if gpu_count > 1 else "auto"

        cmd = [
            sys.executable, "tools/alpha_self_play.py",
            "--engine", str(Path(engine.exe_path).resolve()),
            "--model", model_str,
            "--games", str(n),
            "--simulations", simulations,
            "--batch-size", batch_size,
            "--movetime", "1000",
            "--output", str(data_path),
            "--temperature-drop", str(temperature_drop),
        ]
        if device != "auto":
            cmd += ["--device", device]

        print(f"    worker {w}: {n} games on {device}", file=sys.stderr)
        proc = subprocess.Popen(cmd)
        procs.append(proc)

    for proc in procs:
        proc.wait()

    merged_path = data_dir / f"selfplay_{iteration:04d}.npz"
    if workers > 1:
        _merge_alpha_npz(data_paths, merged_path)
        for p in data_paths:
            if p != merged_path and p.exists():
                p.unlink()
    elif data_paths and data_paths[0].exists():
        data_paths[0].rename(merged_path)

    if not merged_path.exists():
        return False

    return train_and_gate_alpha(
        work_dir, engine, iteration,
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )


def _mcts_selfplay_worker(
    worker_id: int,
    game_indices: list[int],
    exe_path: str,
    options: dict,
    movetime: int,
) -> tuple[list[dict], list[str]]:
    """Worker: play MCTS self-play games with a dedicated engine pair."""
    e1 = ArenaEngine(f"mcts_sp{worker_id}a", exe_path, "mcts", dict(options))
    e2 = ArenaEngine(f"mcts_sp{worker_id}b", exe_path, "mcts", dict(options))
    e1.start()
    e2.start()
    positions: list[dict] = []
    summaries: list[str] = []
    try:
        for i, gi in enumerate(game_indices):
            record = play_arena_game(e1, e2, movetime)
            positions.extend(record["mcts_positions"])
            summaries.append(
                f"  MCTS SP w{worker_id} game {gi+1}: {record['result']} "
                f"({record['ply_count']}mv, {len(record['mcts_positions'])} pos)")
    finally:
        e1.quit()
        e2.quit()
    return positions, summaries


def self_play_and_train_mcts(
    work_dir: Path,
    engine: ArenaEngine,
    iteration: int,
    num_games: int,
    movetime: int,
    gate_games: int,
    gate_movetime: int,
    gate_threshold: float,
    gate_parallel: int = 1,
    opening_suite: list[str] | None = None,
    gpu_count: int = 1,
) -> bool:
    """Run MCTS self-play with parallel GPU workers, train, and gate."""
    data_dir = work_dir / "mcts" / "data"
    model_dir = work_dir / "mcts" / "models"

    best_onnx = model_dir / "best.onnx"
    if not best_onnx.exists():
        print("  MCTS self-play: no best model, skipping", file=sys.stderr)
        return False

    workers = max(1, min(num_games, gpu_count))
    games_per_worker = (num_games + workers - 1) // workers

    chunks: list[list[int]] = []
    for w in range(workers):
        start = w * games_per_worker
        end = min(start + games_per_worker, num_games)
        if start < end:
            chunks.append(list(range(start, end)))

    print(f"  MCTS self-play: {num_games} games, {len(chunks)} workers",
          file=sys.stderr)

    all_positions: list[dict] = []

    if len(chunks) == 1:
        positions, summaries = _mcts_selfplay_worker(
            0, chunks[0], engine.exe_path, engine.options, movetime)
        all_positions.extend(positions)
        for s in summaries:
            print(s, file=sys.stderr)
    else:
        worker_opts: list[dict] = []
        for w in range(len(chunks)):
            opts = dict(engine.options)
            if gpu_count > 1:
                opts["NNDevice"] = f"cuda:{w}"
            worker_opts.append(opts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(
                    _mcts_selfplay_worker, w, chunk,
                    engine.exe_path, worker_opts[w], movetime,
                ): w
                for w, chunk in enumerate(chunks)
            }
            for fut in concurrent.futures.as_completed(futures):
                w = futures[fut]
                try:
                    positions, summaries = fut.result()
                    all_positions.extend(positions)
                    for s in summaries:
                        print(s, file=sys.stderr)
                except Exception as e:
                    print(f"  MCTS SP worker {w} error: {e}", file=sys.stderr)

    if not all_positions:
        print("  MCTS self-play: no positions collected", file=sys.stderr)
        return False

    data_path = str(data_dir / f"selfplay_{iteration:04d}.npz")
    save_mcts_data(all_positions, {}, data_path)
    print(f"  MCTS self-play: saved {len(all_positions)} positions", file=sys.stderr)

    return train_and_gate_mcts(
        work_dir, engine, iteration,
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )


# ---------------------------------------------------------------------------
# Intensive training for weak engines (Feature B)
# ---------------------------------------------------------------------------

def distill_for_weak_engines(
    work_dir: Path,
    engines: dict[str, ArenaEngine],
    elo: EloTracker,
    iteration: int,
    game_records: list[dict],
    gate_games: int,
    gate_movetime: int,
    gate_threshold: float,
    gate_parallel_cpu: int,
    opening_suite: list[str] | None = None,
    min_positions: int = 500,
) -> dict[str, str]:
    """Intensively train weak engines using strongest engine's evaluations.

    Collects ALL positions from game records where the strongest engine
    provided evaluations, then trains each weak engine.
    """
    trained = {}
    names = list(engines.keys())
    if len(names) < 2:
        return trained

    teacher = elo.strongest()
    if teacher is None:
        return trained

    ratings = elo.snapshot()
    median = sorted(ratings.values())[len(ratings) // 2]
    weak_engines = [n for n in names if ratings.get(n, 1500) < median and n != teacher]
    if not weak_engines:
        return trained

    teacher_samples = []
    for record in game_records:
        teacher_samples.extend(
            collect_distillation_samples(record, teacher))

    if len(teacher_samples) < min_positions:
        print(f"  Distill: only {len(teacher_samples)} teacher samples "
              f"(need {min_positions}), skipping", file=sys.stderr)
        return trained

    print(f"  Distill: {len(teacher_samples)} positions from {teacher}, "
          f"training weak engines: {weak_engines}", file=sys.stderr)

    for name in weak_engines:
        eng = engines[name]
        if eng.engine_type == "classic":
            promoted = train_and_gate_classic(
                work_dir, eng, iteration, list(teacher_samples),
                gate_games, gate_movetime, gate_threshold,
                gate_parallel_cpu, opening_suite,
            )
            trained[name] = "promoted" if promoted else "rejected"

        elif eng.engine_type == "nnue":
            nnue_distill = _distill_to_nnue(game_records, teacher)
            if len(nnue_distill) >= min_positions:
                data_path = str(work_dir / "nnue" / "data" / f"distill_{iteration:04d}.npz")
                n = save_nnue_data(nnue_distill, data_path)
                if n > 0:
                    promoted = train_and_gate_nnue(
                        work_dir, eng, iteration,
                        gate_games, gate_movetime, gate_threshold,
                        gate_parallel_cpu, opening_suite,
                    )
                    trained[name] = "promoted" if promoted else "rejected"

    return trained


def _distill_to_nnue(
    game_records: list[dict],
    teacher_name: str,
    eval_scale: float = 361.0,
) -> list[dict]:
    """Build NNUE training samples from teacher engine's evaluations."""
    samples = []
    for record in game_records:
        for p in record["imitation_plies"]:
            if p["mover"] != teacher_name or p["cp"] is None:
                continue
            if not p["bf"] or not p["wf"]:
                continue
            cp = p["cp"]
            label = 1.0 / (1.0 + math.exp(-cp / eval_scale))
            label = max(0.01, min(0.99, label))
            samples.append({
                "black_feats": np.array(p["bf"], dtype=np.int32),
                "white_feats": np.array(p["wf"], dtype=np.int32),
                "side_black": p["side_black"],
                "soft_label": float(label),
                "ply": p["ply"],
            })
    return samples


# ---------------------------------------------------------------------------
# NNUE self-play with MultiPV + softmax
# ---------------------------------------------------------------------------

def _nnue_go_multipv(
    engine: ArenaEngine,
    pos_cmd: str,
    movetime: int,
) -> list[tuple[str, int]]:
    """Send go and parse MultiPV results.

    Returns list of (move, cp) sorted by multipv index, from deepest depth.
    """
    with engine._lock:
        engine._lines.clear()
    engine._send(pos_cmd)
    engine._send(f"go movetime {movetime}")
    lines = engine._wait_for("bestmove", timeout=movetime // 1000 + MOVE_TIMEOUT)

    candidates: dict[int, tuple[str, int]] = {}
    best_depth = 0
    bestmove = "resign"
    last_cp = None

    for line in lines:
        if line.startswith("bestmove"):
            parts = line.split()
            bestmove = parts[1] if len(parts) >= 2 else "resign"
        elif line.startswith("info") and "score" in line:
            m_depth = re.search(r"depth (\d+)", line)
            m_mpv = re.search(r"multipv (\d+)", line)
            m_pv = re.search(r" pv (\S+)", line)
            m_cp = re.search(r"score cp (-?\d+)", line)
            m_mate = re.search(r"score mate (-?\d+)", line)

            cp = None
            if m_cp:
                cp = int(m_cp.group(1))
            elif m_mate:
                cp = 30000 if int(m_mate.group(1)) > 0 else -30000

            if cp is not None:
                last_cp = cp

            if not m_mpv or not m_pv or cp is None:
                continue

            depth = int(m_depth.group(1)) if m_depth else 0
            mpv_idx = int(m_mpv.group(1))
            move = m_pv.group(1)

            if depth > best_depth:
                best_depth = depth
                candidates.clear()
            if depth == best_depth:
                candidates[mpv_idx] = (move, cp)

    if not candidates:
        if bestmove not in ("resign", "none", None):
            return [(bestmove, last_cp if last_cp is not None else 0)]
        return []

    return [(move, cp) for _, (move, cp) in sorted(candidates.items())]


def _softmax_select(
    candidates: list[tuple[str, int]],
    temperature: float = 1.5,
) -> tuple[str, int]:
    """Select a move from candidates using softmax over cp values."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ("resign", 0)
    cps = np.array([cp for _, cp in candidates], dtype=np.float64)
    logits = cps / (temperature * 100.0)
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


_NNUE_OPENING_BLACK = ["7g7f", "2g2f", "5g5f", "6g6f", "3g3f"]
_NNUE_OPENING_WHITE = ["3c3d", "8c8d", "5c5d", "4c4d", "7c7d"]


def _play_nnue_selfplay_game(
    engine: ArenaEngine,
    movetime: int,
    multipv: int = 3,
    temperature: float = 1.5,
    random_plies: int = 4,
) -> list[dict]:
    """Play one NNUE self-play game with MultiPV + softmax move selection."""
    engine.new_game()
    board = make_startpos()
    moves: list[str] = []
    samples: list[dict] = []

    for ply in range(MAX_MOVES):
        pos_cmd = "position startpos"
        if moves:
            pos_cmd += " moves " + " ".join(moves)

        if ply < random_plies:
            if ply == 0:
                bestmove = random.choice(_NNUE_OPENING_BLACK)
            elif ply == 1:
                bestmove = random.choice(_NNUE_OPENING_WHITE)
            else:
                candidates = _nnue_go_multipv(engine, pos_cmd, 50)
                if not candidates:
                    break
                bestmove, _ = _softmax_select(candidates, temperature=3.0)
            if bestmove in ("resign", "none"):
                break
            try:
                apply_usi_move(board, bestmove)
            except Exception:
                break
            moves.append(bestmove)
            continue

        candidates = _nnue_go_multipv(engine, pos_cmd, movetime)
        if not candidates:
            break

        chosen_move, chosen_cp = _softmax_select(candidates, temperature)

        if chosen_move in ("resign", "none"):
            break
        if chosen_cp is not None and chosen_cp <= RESIGN_THRESHOLD:
            break

        bf, wf = extract_features_from_board(board)
        if bf and wf and chosen_cp is not None:
            samples.append({
                "black_feats": np.array(bf, dtype=np.int32),
                "white_feats": np.array(wf, dtype=np.int32),
                "side_black": board.side == 1,
                "soft_label": float(cp_to_label(chosen_cp, False)),
                "ply": ply,
            })

        try:
            apply_usi_move(board, chosen_move)
        except Exception:
            break
        moves.append(chosen_move)

        if detect_repetition(moves):
            break

    return samples


def _nnue_selfplay_worker(
    worker_id: int,
    game_indices: list[int],
    exe_path: str,
    options: dict,
    movetime: int,
    multipv: int,
    temperature: float,
) -> list[dict]:
    """Worker: play several NNUE self-play games with a dedicated engine."""
    engine = ArenaEngine(f"nnue_sp_{worker_id}", exe_path, "nnue", dict(options))
    engine.start()
    all_samples: list[dict] = []
    try:
        for i, gi in enumerate(game_indices):
            samples = _play_nnue_selfplay_game(
                engine, movetime, multipv, temperature)
            all_samples.extend(samples)
            if (i + 1) % 5 == 0 or i == len(game_indices) - 1:
                print(f"  [NNUE SP w{worker_id}] {i+1}/{len(game_indices)} games, "
                      f"{len(all_samples)} samples", file=sys.stderr)
    finally:
        engine.quit()
    return all_samples


def self_play_and_train_nnue(
    work_dir: Path,
    engine: ArenaEngine,
    iteration: int,
    num_games: int,
    movetime: int,
    gate_games: int,
    gate_movetime: int,
    gate_threshold: float,
    gate_parallel: int = 1,
    opening_suite: list[str] | None = None,
    cpu_cores: int = 4,
    multipv: int = 3,
    temperature: float = 1.5,
) -> bool:
    """Run NNUE self-play with MultiPV+softmax, train, and gate."""
    data_dir = work_dir / "nnue" / "data"
    model_dir = work_dir / "nnue" / "models"

    best_bin = model_dir / "best.bin"
    if not best_bin.exists():
        print("  NNUE self-play: no best model, skipping", file=sys.stderr)
        return False

    nnue_threads = int(engine.options.get("Threads", "1"))
    total_hash = int(engine.options.get("Hash", "256"))
    workers = max(1, min(num_games, cpu_cores // max(2, nnue_threads * 2)))
    hash_per_worker = max(16, total_hash // workers)

    opts = dict(engine.options)
    opts["MultiPV"] = str(multipv)
    opts["Hash"] = str(hash_per_worker)
    opts["Threads"] = str(nnue_threads)

    games_per_worker = (num_games + workers - 1) // workers
    chunks: list[list[int]] = []
    for w in range(workers):
        start = w * games_per_worker
        end = min(start + games_per_worker, num_games)
        if start < end:
            chunks.append(list(range(start, end)))

    print(f"  NNUE self-play: {num_games} games, {len(chunks)} workers "
          f"(hash={hash_per_worker}MB/worker, MultiPV={multipv}, "
          f"temp={temperature})", file=sys.stderr)

    all_samples: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _nnue_selfplay_worker, w, chunk,
                engine.exe_path, opts, movetime, multipv, temperature,
            ): w
            for w, chunk in enumerate(chunks)
        }
        for fut in concurrent.futures.as_completed(futures):
            w = futures[fut]
            try:
                samples = fut.result()
                all_samples.extend(samples)
                print(f"  NNUE SP worker {w}: {len(samples)} samples",
                      file=sys.stderr)
            except Exception as e:
                print(f"  NNUE SP worker {w} error: {e}", file=sys.stderr)

    if not all_samples:
        print("  NNUE self-play: no samples collected", file=sys.stderr)
        return False

    data_path = str(data_dir / f"selfplay_{iteration:04d}.npz")
    save_nnue_data(all_samples, data_path)
    print(f"  NNUE self-play: saved {len(all_samples)} samples", file=sys.stderr)

    return train_and_gate_nnue(
        work_dir, engine, iteration,
        gate_games, gate_movetime, gate_threshold, gate_parallel, opening_suite,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "engines": {},
    "arena": {
        "iterations": 50,
        "games_per_round": 20,
        "movetime": 500,
        "parallel": 1,
        "opening_suite": None,
        "work_dir": "arena_work",
        "elo_gap_threshold": 200.0,
    },
    "gate": {
        "threshold": 0.55,
        "games": 40,
        "movetime": 500,
    },
    "selfplay": {
        "games": 20,
        "movetime": 500,
        "temperature_drop": 60,
        "multipv": 3,
        "temperature": 1.5,
    },
    "training": {
        "min_positions": 5000,
        "imitation_elo_gap": 200.0,
        "intensive": False,
    },
}

ENGINE_DEFAULTS: dict[str, dict] = {
    "alpha": {"simulations": 400, "batch_size": 16, "device": "auto"},
    "nnue": {"hash": 256, "threads": 1},
    "mcts": {"simulations": 800, "device": "auto"},
    "classic": {"threads": 1},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    """Load YAML config file and merge with defaults."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    for name, ecfg in list(cfg["engines"].items()):
        if name in ENGINE_DEFAULTS:
            cfg["engines"][name] = _deep_merge(ENGINE_DEFAULTS[name], ecfg)
    return cfg


def build_engines(cfg: dict) -> dict[str, ArenaEngine]:
    """Build ArenaEngine instances from config."""
    engines: dict[str, ArenaEngine] = {}

    for name, ecfg in cfg["engines"].items():
        exe = ecfg.get("exe")
        if not exe:
            continue

        if name == "alpha":
            opts: dict[str, str] = {
                "MctsSimulations": str(ecfg.get("simulations", 400)),
                "MctsBatchSize": str(ecfg.get("batch_size", 16)),
                "Book": "false",
            }
            if ecfg.get("model"):
                opts["NNModel"] = ecfg["model"]
            dev = ecfg.get("device", "auto")
            if dev:
                opts["NNDevice"] = dev
            engines["alpha"] = ArenaEngine("alpha", exe, "alpha", opts)

        elif name == "nnue":
            opts = {
                "Hash": str(ecfg.get("hash", 256)),
                "Threads": str(ecfg.get("threads", 1)),
                "Book": "false",
            }
            if ecfg.get("weights"):
                opts["NNUEFile"] = ecfg["weights"]
            engines["nnue"] = ArenaEngine("nnue", exe, "nnue", opts)

        elif name == "mcts":
            opts = {
                "MctsSimulations": str(ecfg.get("simulations", 800)),
                "Book": "false",
            }
            if ecfg.get("model"):
                opts["NNModel"] = ecfg["model"]
            dev = ecfg.get("device", "auto")
            if dev:
                opts["NNDevice"] = dev
            engines["mcts"] = ArenaEngine("mcts", exe, "mcts", opts)

        elif name == "classic":
            opts = {
                "Book": "false",
                "Threads": str(ecfg.get("threads", 1)),
            }
            if ecfg.get("weights"):
                opts["WeightsFile"] = str(Path(ecfg["weights"]).resolve())
            if ecfg.get("mlp_weights"):
                opts["MlpWeightsFile"] = str(Path(ecfg["mlp_weights"]).resolve())
            engines["classic"] = ArenaEngine("classic", exe, "classic", opts)

    return engines


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-engine arena training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All settings are configured via YAML config file. See arena.yaml.example.",
    )
    parser.add_argument("--config", default="arena.yaml",
                        help="YAML config file (default: arena.yaml)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iterations count")
    parser.add_argument("--clean-start", action="store_true",
                        help="Delete work-dir and start fresh (keep best.* models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        print(f"Create one from arena.yaml.example or specify --config <path>",
              file=sys.stderr)
        return 1

    cfg = load_config(config_path)

    if args.iterations is not None:
        cfg["arena"]["iterations"] = args.iterations

    if args.dry_run:
        yaml.dump(cfg, sys.stdout, default_flow_style=False, allow_unicode=True)
        return 0

    engines = build_engines(cfg)

    if len(engines) < 2:
        print("Error: at least 2 engines required in config 'engines' section.",
              file=sys.stderr)
        return 1

    a = cfg["arena"]
    g = cfg["gate"]
    sp = cfg["selfplay"]
    tr = cfg["training"]

    work_dir = Path(a["work_dir"])
    if args.clean_start and work_dir.exists():
        best_files: list[tuple[Path, bytes]] = []
        for model_dir in work_dir.glob("*/models"):
            for f in model_dir.glob("best.*"):
                best_files.append((f.relative_to(work_dir), f.read_bytes()))
        shutil.rmtree(work_dir)
        for rel, data in best_files:
            dest = work_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        print(f"Cleaned work directory: {work_dir} (kept {len(best_files)} best.* files)", file=sys.stderr)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "games").mkdir(exist_ok=True)
    for etype in ("alpha", "nnue", "mcts", "classic"):
        (work_dir / etype / "data").mkdir(parents=True, exist_ok=True)
        (work_dir / etype / "models").mkdir(parents=True, exist_ok=True)

    elo = EloTracker(list(engines.keys()))
    elo_path = str(work_dir / "elo.json")
    elo.load(elo_path)

    log_path = work_dir / "train_log.jsonl"

    engine_names = list(engines.keys())
    n_parallel = max(1, a.get("parallel", 1))

    cpu_cores, gpu_count = detect_hardware()

    gate_parallel_gpu = max(1, min(gpu_count * 2, 4))
    gate_parallel_cpu = max(1, min(cpu_cores // 4, 8))

    print(f"Arena: {', '.join(engine_names)}", file=sys.stderr)
    print(f"  iterations={a['iterations']}, games/round={a['games_per_round']}, "
          f"movetime={a['movetime']}ms, parallel={n_parallel}", file=sys.stderr)
    print(f"  gate: {g['games']} games, threshold={g['threshold']}, "
          f"parallel=gpu:{gate_parallel_gpu}/cpu:{gate_parallel_cpu}", file=sys.stderr)
    print(f"  work_dir={work_dir}", file=sys.stderr)

    opening_suite: list[str] | None = None
    if a.get("opening_suite"):
        suite_path = Path(a["opening_suite"])
        if suite_path.is_file():
            opening_suite = [l.strip() for l in suite_path.read_text().splitlines() if l.strip()]
            print(f"  opening_suite: {len(opening_suite)} positions from {suite_path}",
                  file=sys.stderr)
        else:
            print(f"  WARNING: opening suite not found: {suite_path}", file=sys.stderr)

    engine_pools: list[dict[str, ArenaEngine]] = []
    for slot in range(n_parallel):
        slot_engines: dict[str, ArenaEngine] = {}
        for name, tmpl in engines.items():
            e = ArenaEngine(name, tmpl.exe_path, tmpl.engine_type, dict(tmpl.options))
            e.start()
            slot_engines[name] = e
        engine_pools.append(slot_engines)

    alpha_all_positions: list[list[dict]] = []
    alpha_all_results: list[str] = []
    nnue_all_samples: list[dict] = []
    mcts_all_positions: list[dict] = []
    classic_distill_samples: list[tuple[str, str, float]] = []
    iter_game_records: list[dict] = []
    game_counter = 0
    data_lock = threading.Lock()

    try:
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
        from rich.panel import Panel
        from rich.layout import Layout
        has_rich = True
    except ImportError:
        has_rich = False

    slot_status: list[dict] = [{"state": "idle"} for _ in range(n_parallel)]
    live_ctx = [None]  # mutable ref for nested functions
    completed_games: list[str] = []  # recent results log
    h2h: dict[tuple[str, str], list[int]] = {}
    for a in engine_names:
        for b in engine_names:
            if a != b:
                h2h[(a, b)] = [0, 0, 0]  # [wins, losses, draws]

    def _build_display():
        table = Table(title="Arena", show_header=True, expand=True, padding=(0, 1))
        table.add_column("Slot", style="bold", width=4, justify="center")
        table.add_column("Matchup", width=22)
        table.add_column("Ply", width=5, justify="right")
        table.add_column("B Eval", width=8, justify="right")
        table.add_column("W Eval", width=8, justify="right")
        table.add_column("Status", width=12)
        for i, ss in enumerate(slot_status):
            if ss["state"] == "idle":
                table.add_row(str(i+1), ss.get("matchup", "-"), "-", "-", "-",
                              Text("idle", style="dim"))
            elif ss["state"] == "done":
                style = "green" if "win" in ss.get("result_detail", "") else ("yellow" if "draw" in ss.get("result_detail", "") else "red")
                table.add_row(str(i+1), ss.get("matchup", ""),
                              str(ss.get("ply", "")),
                              ss.get("b_eval", ""), ss.get("w_eval", ""),
                              Text(ss.get("result_detail", ""), style=style))
            else:
                matchup = ss.get("matchup", "")
                table.add_row(str(i+1), matchup,
                              str(ss.get("ply", 0)),
                              ss.get("b_eval", ""), ss.get("w_eval", ""),
                              Text("playing", style="cyan"))

        elo_parts = []
        for name in engine_names:
            r = elo.ratings.get(name, 1500)
            elo_parts.append(f"{name}:{r:.0f}")
        elo_line = "  ".join(elo_parts)

        recent = "\n".join(completed_games[-n_parallel*2:]) if completed_games else "(no results yet)"
        log_panel = Panel(recent, title="Recent", border_style="dim", expand=True)

        matrix = Table(title="Head-to-Head", show_header=True, padding=(0, 1),
                       expand=True)
        matrix.add_column("", style="bold", width=10)
        for name in engine_names:
            matrix.add_column(name, width=10, justify="center")
        for row_name in engine_names:
            cells = []
            for col_name in engine_names:
                if row_name == col_name:
                    cells.append("-")
                else:
                    w, l, d = h2h.get((row_name, col_name), [0, 0, 0])
                    if w + l + d == 0:
                        cells.append("-")
                    else:
                        cells.append(f"{w} - {l}")
            matrix.add_row(row_name, *cells)

        layout = Layout()
        layout.split_column(
            Layout(Panel(elo_line, title="Elo Ratings", border_style="blue"), size=3),
            Layout(table, size=n_parallel + 4),
            Layout(log_panel, size=min(n_parallel * 2 + 3, 10)),
            Layout(matrix, size=len(engine_names) + 4),
        )
        return layout

    def _refresh():
        if has_rich and live_ctx[0]:
            live_ctx[0].update(_build_display())

    shuffled_arena_suite: list[str] = []
    if opening_suite:
        shuffled_arena_suite = list(opening_suite)
        random.shuffle(shuffled_arena_suite)

    def _play_one(slot_idx: int, game_idx: int, iteration: int,
                  black_name: str, white_name: str,
                  opening: list[str] | None = None) -> dict | None:
        slot = engine_pools[slot_idx]
        black_engine = slot[black_name]
        white_engine = slot[white_name]

        matchup_str = f"{black_name}(B) vs {white_name}(W)"
        slot_status[slot_idx] = {
            "state": "playing", "matchup": matchup_str,
            "ply": 0, "b_eval": "-", "w_eval": "-",
        }
        _refresh()

        def on_move(ply, move, cp, winner):
            eval_str = f"{cp:+d}" if cp is not None else "?"
            is_black_move = (ply % 2 == 1)
            eval_key = "b_eval" if is_black_move else "w_eval"
            if winner:
                w = black_name if winner == "black" else white_name
                slot_status[slot_idx].update({
                    "state": "done", "ply": ply,
                    eval_key: eval_str,
                    "result_detail": f"{w} win",
                })
            else:
                slot_status[slot_idx].update({
                    "ply": ply,
                    eval_key: eval_str,
                })
            _refresh()

        try:
            return play_arena_game(black_engine, white_engine, a["movetime"],
                                   on_move, opening)
        except (OSError, BrokenPipeError) as e:
            slot_status[slot_idx] = {"state": "idle"}
            _refresh()
            black_engine.restart()
            if black_name != white_name:
                white_engine.restart()
            return None

    live_manager = Live(_build_display(), refresh_per_second=4, console=Console(stderr=True)) if has_rich else None

    try:
        if live_manager:
            live_manager.start()
            live_ctx[0] = live_manager

        for iteration in range(a["iterations"]):
            iter_start = time.time()
            round_results: list[dict] = []

            if not has_rich:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"  Iteration {iteration + 1}/{a['iterations']}  "
                      f"Elo: {elo.snapshot()}", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)

            for s in slot_status:
                s.update({"state": "idle"})
            for key in h2h:
                h2h[key] = [0, 0, 0]
            iter_game_records.clear()
            _refresh()

            free_slots = list(range(n_parallel))
            games_started = 0
            games_done = 0
            pair_game_counts: dict[tuple[str, str], int] = {}
            for a in engine_names:
                for b in engine_names:
                    if a < b:
                        pair_game_counts[(a, b)] = 0
            pending: dict[concurrent.futures.Future, tuple[int, int, str, str]] = {}
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel)

            def _submit_game():
                nonlocal games_started
                if games_started >= a["games_per_round"] or not free_slots:
                    return
                slot_idx = free_slots.pop(0)
                gi = games_started
                black_name, white_name = select_matchup(
                    engine_names, elo, a["elo_gap_threshold"], pair_game_counts)
                pair_key = (min(black_name, white_name), max(black_name, white_name))
                pair_count = pair_game_counts.get(pair_key, 0)
                if pair_count % 2 == 1 and black_name != white_name:
                    black_name, white_name = white_name, black_name
                opening = None
                if shuffled_arena_suite:
                    oi = pair_count // 2
                    opening = shuffled_arena_suite[oi % len(shuffled_arena_suite)].split()
                pair_game_counts[pair_key] = pair_count + 1
                fut = executor.submit(_play_one, slot_idx, gi, iteration,
                                      black_name, white_name, opening)
                pending[fut] = (slot_idx, gi, black_name, white_name)
                games_started += 1

            for _ in range(min(n_parallel, a["games_per_round"])):
                _submit_game()

            while games_done < a["games_per_round"] and pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    slot_idx, gi, bn, wn = pending.pop(future)
                    free_slots.append(slot_idx)
                    games_done += 1
                    record = future.result()

                    _submit_game()
                    if slot_idx in free_slots:
                        slot_status[slot_idx] = {"state": "idle", "matchup": "---"}
                        _refresh()

                    if record is None:
                        continue

                    with data_lock:
                        game_counter += 1

                        result = record["result"]
                        if result == "black":
                            elo_result = 1.0
                        elif result == "white":
                            elo_result = 0.0
                        else:
                            elo_result = 0.5
                        elo.update(bn, wn, elo_result)

                        if result == "black":
                            h2h[(bn, wn)][0] += 1
                            h2h[(wn, bn)][1] += 1
                        elif result == "white":
                            h2h[(wn, bn)][0] += 1
                            h2h[(bn, wn)][1] += 1
                        else:
                            h2h[(bn, wn)][2] += 1
                            h2h[(wn, bn)][2] += 1

                        if record["alpha_positions"]:
                            alpha_all_positions.append(record["alpha_positions"])
                            alpha_all_results.append(record["result"])
                        if record["nnue_samples"]:
                            nnue_all_samples.extend(record["nnue_samples"])
                        if record["mcts_positions"]:
                            mcts_all_positions.extend(record["mcts_positions"])

                        if result in ("black", "white") and tr["imitation_elo_gap"] > 0:
                            winner_name = bn if result == "black" else wn
                            loser_name = wn if result == "black" else bn
                            gap = elo.ratings.get(winner_name, 1500) - elo.ratings.get(loser_name, 1500)
                            if gap >= tr["imitation_elo_gap"]:
                                loser_type = engines[loser_name].engine_type
                                if loser_type != "classic":
                                    _imitation_for_loser(
                                        record["imitation_plies"], winner_name, loser_type,
                                        alpha_all_positions, alpha_all_results,
                                        nnue_all_samples, mcts_all_positions,
                                    )

                        if "classic" in engines:
                            teacher = elo.strongest()
                            if teacher != "classic":
                                classic_distill_samples.extend(
                                    collect_distillation_samples(record, teacher))

                        winner = bn if result == "black" else (wn if result == "white" else "draw")
                        log_line = f"[{iteration+1}.{gi+1}] {bn} vs {wn} -> {winner} ({record['ply_count']}mv)"
                        completed_games.append(log_line)

                        if not has_rich:
                            print(f"  {log_line}  Elo: {elo.snapshot()}", file=sys.stderr)

                        _refresh()

                        game_path = work_dir / "games" / f"game_{game_counter:05d}.json"
                        game_log = {
                            "black": record["black"], "white": record["white"],
                            "result": record["result"], "ply_count": record["ply_count"],
                            "moves": record["moves"],
                        }
                        with open(game_path, "w") as f:
                            json.dump(game_log, f)

                        round_results.append({
                            "black": record["black"], "white": record["white"],
                            "result": record["result"],
                        })

                        if tr["intensive"]:
                            iter_game_records.append(record)

            executor.shutdown(wait=False)

            elo.save(elo_path)

            data_saved = {}
            if sum(len(p) for p in alpha_all_positions) >= tr["min_positions"]:
                path = str(work_dir / "alpha" / "data" / f"iter_{iteration+1:04d}.npz")
                n = save_alpha_data(alpha_all_positions, alpha_all_results, path)
                if n > 0:
                    data_saved["alpha"] = n
                    alpha_all_positions.clear()
                    alpha_all_results.clear()

            if len(nnue_all_samples) >= tr["min_positions"]:
                path = str(work_dir / "nnue" / "data" / f"iter_{iteration+1:04d}.npz")
                n = save_nnue_data(nnue_all_samples, path)
                if n > 0:
                    data_saved["nnue"] = n
                    nnue_all_samples.clear()

            if len(mcts_all_positions) >= tr["min_positions"]:
                path = str(work_dir / "mcts" / "data" / f"iter_{iteration+1:04d}.npz")
                n = save_mcts_data(mcts_all_positions, {}, path)
                if n > 0:
                    data_saved["mcts"] = n
                    mcts_all_positions.clear()

            if has_rich and live_manager:
                live_manager.stop()
                live_ctx[0] = None

            trained = {}
            if "alpha" in data_saved and "alpha" in engines:
                print(f"  Training alpha...", file=sys.stderr)
                promoted = train_and_gate_alpha(
                    work_dir, engines["alpha"], iteration + 1,
                    g["games"], g["movetime"], g["threshold"],
                    gate_parallel_gpu, opening_suite)
                trained["alpha"] = "promoted" if promoted else "rejected"

            if "nnue" in data_saved and "nnue" in engines:
                print(f"  Training nnue...", file=sys.stderr)
                promoted = train_and_gate_nnue(
                    work_dir, engines["nnue"], iteration + 1,
                    g["games"], g["movetime"], g["threshold"],
                    gate_parallel_cpu, opening_suite)
                trained["nnue"] = "promoted" if promoted else "rejected"

            if "mcts" in data_saved and "mcts" in engines:
                print(f"  Training mcts...", file=sys.stderr)
                promoted = train_and_gate_mcts(
                    work_dir, engines["mcts"], iteration + 1,
                    g["games"], g["movetime"], g["threshold"],
                    gate_parallel_gpu, opening_suite)
                trained["mcts"] = "promoted" if promoted else "rejected"

            if ("classic" in engines
                    and len(classic_distill_samples) >= tr["min_positions"]):
                print(f"  Training classic (distillation, "
                      f"{len(classic_distill_samples)} samples)...",
                      file=sys.stderr)
                promoted = train_and_gate_classic(
                    work_dir, engines["classic"], iteration + 1,
                    list(classic_distill_samples),
                    g["games"], g["movetime"], g["threshold"],
                    gate_parallel_cpu, opening_suite)
                trained["classic"] = "promoted" if promoted else "rejected"
                classic_distill_samples.clear()

            # --- Feature A: Self-play for top engines ---
            if sp["games"] > 0:
                ratings = elo.snapshot()
                median_elo = sorted(ratings.values())[len(ratings) // 2]
                for name in engine_names:
                    if ratings.get(name, 1500) < median_elo:
                        continue
                    if name in trained:
                        continue
                    eng = engines[name]
                    if eng.engine_type == "alpha":
                        print(f"  Self-play alpha ({sp['games']} games)...",
                              file=sys.stderr)
                        promoted = self_play_and_train_alpha(
                            work_dir, eng, iteration + 1,
                            sp["games"],
                            g["games"], g["movetime"],
                            g["threshold"], gate_parallel_gpu,
                            opening_suite,
                            gpu_count=gpu_count,
                            temperature_drop=sp["temperature_drop"])
                        trained["alpha_sp"] = "promoted" if promoted else "rejected"
                    elif eng.engine_type == "mcts":
                        print(f"  Self-play mcts ({sp['games']} games)...",
                              file=sys.stderr)
                        promoted = self_play_and_train_mcts(
                            work_dir, eng, iteration + 1,
                            sp["games"], sp["movetime"],
                            g["games"], g["movetime"],
                            g["threshold"], gate_parallel_gpu,
                            opening_suite,
                            gpu_count=gpu_count)
                        trained["mcts_sp"] = "promoted" if promoted else "rejected"
                    elif eng.engine_type == "nnue":
                        print(f"  Self-play nnue ({sp['games']} games, "
                              f"MultiPV={sp['multipv']})...",
                              file=sys.stderr)
                        promoted = self_play_and_train_nnue(
                            work_dir, eng, iteration + 1,
                            sp["games"], sp["movetime"],
                            g["games"], g["movetime"],
                            g["threshold"], gate_parallel_cpu,
                            opening_suite,
                            cpu_cores=cpu_cores,
                            multipv=sp["multipv"],
                            temperature=sp["temperature"])
                        trained["nnue_sp"] = "promoted" if promoted else "rejected"

            # --- Feature B: Intensive distillation for weak engines ---
            if tr["intensive"] and iter_game_records:
                distill_results = distill_for_weak_engines(
                    work_dir, engines, elo, iteration + 1,
                    iter_game_records,
                    g["games"], g["movetime"],
                    g["threshold"], gate_parallel_cpu,
                    opening_suite)
                for k, v in distill_results.items():
                    trained[f"{k}_distill"] = v

            iter_elapsed = time.time() - iter_start
            log_entry = {
                "iteration": iteration + 1,
                "elo": elo.snapshot(),
                "games": round_results,
                "data_saved": data_saved,
                "trained": trained,
                "elapsed_s": round(iter_elapsed, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"\n  Iteration {iteration+1} complete ({iter_elapsed:.0f}s) "
                  f"Elo: {elo.snapshot()}", file=sys.stderr)
            if data_saved:
                print(f"  Data saved: {data_saved}", file=sys.stderr)
            if trained:
                print(f"  Training: {trained}", file=sys.stderr)

            if has_rich and iteration + 1 < a["iterations"]:
                completed_games.clear()
                live_manager = Live(_build_display(), refresh_per_second=4, console=Console(stderr=True))
                live_manager.start()
                live_ctx[0] = live_manager

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
    finally:
        if has_rich and live_ctx[0]:
            live_ctx[0].stop()
        for slot in engine_pools:
            for e in slot.values():
                e.quit()
        elo.save(elo_path)

        if alpha_all_positions:
            path = str(work_dir / "alpha" / "data" / "remaining.npz")
            save_alpha_data(alpha_all_positions, alpha_all_results, path)
        if nnue_all_samples:
            path = str(work_dir / "nnue" / "data" / "remaining.npz")
            save_nnue_data(nnue_all_samples, path)
        if mcts_all_positions:
            path = str(work_dir / "mcts" / "data" / "remaining.npz")
            save_mcts_data(mcts_all_positions, {}, path)
        if classic_distill_samples:
            path = str(work_dir / "classic" / "data" / "remaining.tsv")
            save_distillation_tsv(classic_distill_samples, path)

    print(f"\nFinal Elo: {elo.snapshot()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
