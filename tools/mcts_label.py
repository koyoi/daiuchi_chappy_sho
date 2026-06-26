#!/usr/bin/env python3
"""Label positions with a strong USI engine (MCTS/Alpha/NNUE) for MLP training.

Reads sfen+move pairs, evaluates each position after the move using a USI
engine, and outputs sfen+move+label triples.  The label is win probability
from the mover's perspective.

Usage:
  python tools/mcts_label.py \\
    --input mlp_positions.tsv \\
    --output mlp_positions_labeled.tsv \\
    --engine build/Release/kishi-to-alpha.exe \\
    --movetime 100 --workers 4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


class USIEngine:
    """Manage a USI engine subprocess."""

    def __init__(self, path: str, options: Optional[dict] = None):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("usi")
        self._wait_for("usiok")
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Engine process terminated unexpectedly")
            if line.strip() == token:
                return

    def evaluate_after_move(self, sfen: str, usi_move: str,
                            movetime: int = 100) -> float:
        """Evaluate position after usi_move. Returns WP from mover's perspective."""
        self._send(f"position sfen {sfen} moves {usi_move}")
        self._send(f"go movetime {movetime}")

        score_cp = 0
        is_mate = False
        mate_ply = 0

        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Engine process terminated unexpectedly")
            line = line.strip()
            if line.startswith("bestmove"):
                break
            parts = line.split()
            if "score" not in parts:
                continue
            try:
                si = parts.index("score")
                if parts[si + 1] == "cp":
                    score_cp = int(parts[si + 2])
                    is_mate = False
                elif parts[si + 1] == "mate":
                    mate_ply = int(parts[si + 2])
                    is_mate = True
            except (ValueError, IndexError):
                pass

        if is_mate:
            # From opponent's (current side-to-move) perspective:
            #   mate_ply > 0 = opponent can mate = bad for mover
            #   mate_ply < 0 = opponent is mated = good for mover
            opponent_wp = 0.99 if mate_ply > 0 else 0.01
        else:
            # score_cp is from opponent's perspective
            opponent_wp = 1.0 / (1.0 + math.exp(-score_cp / 361.0))

        mover_wp = 1.0 - opponent_wp
        return max(0.01, min(0.99, mover_wp))

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def label_chunk(engine_path: str, positions: List[Tuple[str, str]],
                movetime: int, options: Optional[dict],
                results: list, offset: int, progress_lock: threading.Lock,
                counter: list):
    """Label a chunk of positions with a single engine instance."""
    engine = USIEngine(engine_path, options)
    for i, (sfen, move) in enumerate(positions):
        try:
            wp = engine.evaluate_after_move(sfen, move, movetime)
        except Exception:
            wp = 0.5
        results[offset + i] = (sfen, move, wp)
        with progress_lock:
            counter[0] += 1
    engine.close()


def main():
    parser = argparse.ArgumentParser(
        description="Label positions with MCTS/Alpha engine evaluation")
    parser.add_argument("--input", required=True,
                        help="Input TSV (sfen<TAB>usi_move per line)")
    parser.add_argument("--output", required=True,
                        help="Output TSV (sfen<TAB>usi_move<TAB>label)")
    parser.add_argument("--engine", required=True,
                        help="USI engine executable (e.g. kishi-to-alpha)")
    parser.add_argument("--movetime", type=int, default=100,
                        help="Movetime in ms per position (default: 100)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel engine instances (default: 1)")
    parser.add_argument("--hash", type=int, default=64,
                        help="Hash table size in MB per engine (default: 64)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max positions to label (0=all)")
    args = parser.parse_args()

    positions = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tab = line.find("\t")
            if tab < 0:
                continue
            sfen = line[:tab]
            usi_move = line[tab + 1:].split("\t")[0]
            positions.append((sfen, usi_move))

    if args.limit > 0:
        positions = positions[:args.limit]

    print(f"Labeling {len(positions)} positions with {args.workers} worker(s), "
          f"movetime={args.movetime}ms", file=sys.stderr)

    engine_options = {"Hash": str(args.hash)}
    results = [None] * len(positions)
    progress_lock = threading.Lock()
    counter = [0]

    # Split positions across workers
    n = len(positions)
    w = min(args.workers, n)
    chunk_size = (n + w - 1) // w

    threads = []
    for i in range(w):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            break
        chunk = positions[start:end]
        t = threading.Thread(
            target=label_chunk,
            args=(args.engine, chunk, args.movetime, engine_options,
                  results, start, progress_lock, counter),
        )
        threads.append(t)
        t.start()

    # Progress reporting
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        with progress_lock:
            done = counter[0]
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"\r  {done}/{n} ({100*done/n:.1f}%) "
              f"{rate:.1f} pos/s, ETA {eta/60:.0f}min",
              end="", file=sys.stderr)

    for t in threads:
        t.join()

    print(f"\r  {n}/{n} (100.0%) done.                    ", file=sys.stderr)

    # Write output
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            if r is None:
                continue
            sfen, move, wp = r
            f.write(f"{sfen}\t{move}\t{wp:.6f}\n")
            written += 1

    print(f"Wrote {written} labeled positions to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
