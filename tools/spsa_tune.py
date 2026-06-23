#!/usr/bin/env python3
"""SPSA (Simultaneous Perturbation Stochastic Approximation) tuner for
kishi-to-nnue search parameters.

Runs self-play games between two engine instances with perturbed parameters,
then adjusts parameters toward the winning perturbation direction.

Usage:
  python tools/spsa_tune.py --engine build/Release/kishi-to-nnue.exe
  python tools/spsa_tune.py --engine build/Release/kishi-to-nnue.exe --iterations 200 --games 24
  python tools/spsa_tune.py --engine build/Release/kishi-to-nnue.exe --resume spsa_state.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Param:
    name: str
    value: float
    min_val: float
    max_val: float
    c_end: float = 0.0    # perturbation size (auto-set if 0)
    a_end: float = 0.0    # step size (auto-set if 0)
    integer: bool = True

    def __post_init__(self):
        if self.c_end <= 0:
            self.c_end = max(1.0, (self.max_val - self.min_val) * 0.05)
        if self.a_end <= 0:
            self.a_end = self.c_end * 2.0


DEFAULT_PARAMS = [
    Param("LMRFullDepthMoves", 4, 1, 20),
    Param("LMRMinDepth",       3, 1, 10),
    Param("NMPMinDepth",       3, 1, 10),
    Param("NMPReduction",      3, 1, 8),
    Param("FutilityMargin1",   400, 50, 2000),
    Param("FutilityMargin2",   900, 100, 3000),
    Param("AspirationWindow",  50, 10, 500),
    Param("IIDMinDepth",       5, 2, 10),
    Param("DeltaMargin",       1400, 200, 5000),
    Param("QDepth",            6, 1, 20),
    Param("QCheckDepthMin",    4, 1, 10),
    Param("RootPruneWidth",    15, 1, 100),
]


class USIEngine:
    """Manages a USI engine subprocess."""

    def __init__(self, exe_path: str, params: Optional[Dict[str, int]] = None,
                 movetime: int = 300, threads: int = 1):
        self.exe_path = os.path.abspath(exe_path)
        self.params = params or {}
        self.movetime = movetime
        self.threads = threads
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("usi")
        self._wait_for("usiok")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name MaxMoveTimeMs value {self.movetime}")
        self._send("setoption name Book value false")
        for name, val in self.params.items():
            self._send(f"setoption name {name} value {val}")
        self._send("isready")
        self._wait_for("readyok")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None

    def new_game(self):
        self._send("usinewgame")

    def go(self, position_cmd: str) -> Optional[str]:
        self._send(position_cmd)
        self._send(f"go movetime {self.movetime}")
        while True:
            line = self._readline()
            if line is None:
                return None
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def _readline(self) -> Optional[str]:
        if self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if line:
                return line.strip()
        return None

    def _wait_for(self, token: str, timeout: float = 30.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self._readline()
            if line and token in line:
                return
        raise TimeoutError(f"Timeout waiting for '{token}'")


def play_game(engine_black: USIEngine, engine_white: USIEngine,
              max_moves: int = 256) -> float:
    """Play a single game. Returns +1 for black win, -1 for white win, 0 for draw."""
    engine_black.new_game()
    engine_white.new_game()

    moves: List[str] = []
    for ply in range(max_moves):
        if moves:
            pos_cmd = "position startpos moves " + " ".join(moves)
        else:
            pos_cmd = "position startpos"

        engine = engine_black if ply % 2 == 0 else engine_white
        bestmove = engine.go(pos_cmd)

        if bestmove is None or bestmove == "resign" or bestmove == "none":
            return -1.0 if ply % 2 == 0 else 1.0

        moves.append(bestmove)

    return 0.0


def play_match(exe_path: str, params_a: Dict[str, int], params_b: Dict[str, int],
               num_games: int, movetime: int, threads: int) -> float:
    """Play a match between params_a and params_b.

    Returns score from A's perspective: wins + 0.5*draws / total.
    Each pair of games swaps colors.
    """
    score_a = 0.0
    pairs = num_games // 2

    for pair in range(pairs):
        eng_a = USIEngine(exe_path, params_a, movetime, threads)
        eng_b = USIEngine(exe_path, params_b, movetime, threads)
        try:
            eng_a.start()
            eng_b.start()

            # Game 1: A=black, B=white
            result = play_game(eng_a, eng_b)
            if result > 0:
                score_a += 1.0
            elif result == 0:
                score_a += 0.5

            # Game 2: B=black, A=white
            result = play_game(eng_b, eng_a)
            if result < 0:
                score_a += 1.0
            elif result == 0:
                score_a += 0.5

        finally:
            eng_a.stop()
            eng_b.stop()

    return score_a / max(num_games, 1)


def spsa_coefficients(k: int, total_iterations: int, a0: float, c0: float
                      ) -> Tuple[float, float]:
    """Compute SPSA step sizes a_k and c_k for iteration k."""
    alpha = 0.602
    gamma = 0.101
    A = total_iterations * 0.1
    a_k = a0 / ((k + 1 + A) ** alpha)
    c_k = c0 / ((k + 1) ** gamma)
    return a_k, c_k


def run_spsa(args):
    params = list(DEFAULT_PARAMS)

    # Load state if resuming
    state_file = args.resume or "spsa_state.json"
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            state = json.load(f)
        for p in params:
            if p.name in state.get("values", {}):
                p.value = state["values"][p.name]
        start_iter = state.get("iteration", 0)
        print(f"Resumed from iteration {start_iter}")

    print(f"SPSA tuning: {len(params)} parameters, {args.iterations} iterations, "
          f"{args.games} games/iter, movetime={args.movetime}ms")
    print(f"Engine: {args.engine}")
    print()

    for p in params:
        print(f"  {p.name:25s} = {p.value:8.1f}  [{p.min_val}, {p.max_val}]  "
              f"c={p.c_end:.1f} a={p.a_end:.1f}")
    print()

    for k in range(start_iter, start_iter + args.iterations):
        # Generate random perturbation direction
        delta = [random.choice([-1, 1]) for _ in params]

        # Compute step sizes
        a_k, c_k = spsa_coefficients(k, start_iter + args.iterations,
                                     args.a_ratio, args.c_ratio)

        # Create perturbed parameter sets
        params_plus: Dict[str, int] = {}
        params_minus: Dict[str, int] = {}
        for i, p in enumerate(params):
            pert = c_k * p.c_end * delta[i]
            v_plus = max(p.min_val, min(p.max_val, p.value + pert))
            v_minus = max(p.min_val, min(p.max_val, p.value - pert))
            params_plus[p.name] = int(round(v_plus))
            params_minus[p.name] = int(round(v_minus))

        # Play match
        t0 = time.time()
        score_plus = play_match(args.engine, params_plus, params_minus,
                                args.games, args.movetime, args.threads)
        elapsed = time.time() - t0

        # Update parameters
        y_diff = score_plus - 0.5  # deviation from expected draw
        for i, p in enumerate(params):
            pert = c_k * p.c_end * delta[i]
            if abs(pert) < 1e-9:
                continue
            gradient = y_diff / (2.0 * pert)
            step = a_k * p.a_end * gradient
            p.value = max(p.min_val, min(p.max_val, p.value + step))

        # Report
        win_pct = score_plus * 100
        print(f"[iter {k+1:4d}] score={win_pct:5.1f}%  "
              f"y_diff={y_diff:+.3f}  a_k={a_k:.4f}  c_k={c_k:.4f}  "
              f"({elapsed:.1f}s)")

        # Save state periodically
        if (k + 1) % args.save_every == 0 or k == start_iter + args.iterations - 1:
            state = {
                "iteration": k + 1,
                "values": {p.name: round(p.value, 2) for p in params},
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            print(f"  State saved to {state_file}")

            # Print current values
            print("  Current values:")
            for p in params:
                val = int(round(p.value)) if p.integer else p.value
                print(f"    {p.name:25s} = {val}")
            print()

    # Final output
    print("\n=== Final optimized parameters ===")
    for p in params:
        val = int(round(p.value)) if p.integer else p.value
        print(f"  setoption name {p.name} value {val}")

    # Save final USI commands file
    usi_file = "spsa_best_params.txt"
    with open(usi_file, "w") as f:
        for p in params:
            val = int(round(p.value)) if p.integer else p.value
            f.write(f"setoption name {p.name} value {val}\n")
    print(f"\nUSI commands saved to {usi_file}")


def main():
    parser = argparse.ArgumentParser(
        description="SPSA search parameter tuner for kishi-to-nnue")
    parser.add_argument("--engine", required=True,
                        help="Path to kishi-to-nnue executable")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of SPSA iterations (default: 100)")
    parser.add_argument("--games", type=int, default=16,
                        help="Games per iteration (must be even, default: 16)")
    parser.add_argument("--movetime", type=int, default=300,
                        help="Time per move in ms (default: 300)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads per engine (default: 1)")
    parser.add_argument("--a-ratio", type=float, default=1.0,
                        help="SPSA step size multiplier (default: 1.0)")
    parser.add_argument("--c-ratio", type=float, default=1.0,
                        help="SPSA perturbation multiplier (default: 1.0)")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save state every N iterations (default: 5)")
    parser.add_argument("--resume", default="",
                        help="Resume from state file")
    args = parser.parse_args()

    if args.games % 2 != 0:
        args.games += 1
        print(f"Adjusted games to {args.games} (must be even for color fairness)")

    if not os.path.exists(args.engine):
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        return 1

    run_spsa(args)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
