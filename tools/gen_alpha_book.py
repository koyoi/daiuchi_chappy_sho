#!/usr/bin/env python3
"""Generate opening book via high-simulation MCTS analysis with the Alpha engine.

Explores opening lines by running the engine with high simulations on each position,
using visit counts as move weights. Produces deeper/more accurate book entries
than the statistics-based gen_book.py.

Usage:
  python gen_alpha_book.py --engine build/Release/kishi-to-alpha.exe --plies 12 --output book.txt
  python gen_alpha_book.py --engine build/Release/kishi-to-alpha.exe --plies 8 --simulations 6400 --branches 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from alpha_self_play import AlphaUSIEngine


def expand_position(engine: AlphaUSIEngine, position_cmd: str,
                    movetime: int) -> list[tuple[str, int]]:
    """Run engine on a position and return [(move, visit_count), ...] sorted by visits."""
    bestmove, visits_raw = engine.go(position_cmd, movetime)
    if bestmove == "resign" or not visits_raw:
        return []

    move_visits = []
    engine._send("getvisits")
    visit_line = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        line = engine._recv()
        if line is None:
            break
        if line.startswith("visits"):
            visit_line = line
            break

    if visit_line:
        parts = visit_line.split()[1:]
        for part in parts:
            tokens = part.split(":")
            if len(tokens) >= 2:
                move_str = tokens[0]
                count = int(tokens[1])
                if count > 0:
                    move_visits.append((move_str, count))

    if not move_visits and bestmove != "resign":
        move_visits = [(bestmove, 1)]

    move_visits.sort(key=lambda x: -x[1])
    return move_visits


def main():
    parser = argparse.ArgumentParser(
        description="Generate opening book via Alpha engine MCTS analysis")
    parser.add_argument("--engine", required=True, help="Path to kishi-to-alpha.exe")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--device", default=None, help="NN device")
    parser.add_argument("--output", default="book.txt", help="Output book file")
    parser.add_argument("--plies", type=int, default=12, help="Max opening depth")
    parser.add_argument("--simulations", type=int, default=6400,
                        help="MCTS simulations per position")
    parser.add_argument("--movetime", type=int, default=10000, help="Time per position (ms)")
    parser.add_argument("--branches", type=int, default=3,
                        help="Top moves to expand per position")
    parser.add_argument("--min-visit-ratio", type=float, default=0.05,
                        help="Min visit ratio to include a move")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing book instead of overwriting")
    args = parser.parse_args()

    engine = AlphaUSIEngine(
        args.engine, model=args.model, device=args.device,
        simulations=args.simulations, temperature_drop=0,
    )
    engine.start()

    book_entries: dict[str, list[tuple[str, int]]] = {}

    queue: list[tuple[str, list[str]]] = [("position startpos", [])]
    processed = 0
    t0 = time.time()

    print(f"Generating book: depth={args.plies}, sims={args.simulations}, "
          f"branches={args.branches}", file=sys.stderr)

    try:
        while queue:
            pos_cmd, moves_so_far = queue.pop(0)
            depth = len(moves_so_far)

            if depth >= args.plies:
                continue

            processed += 1
            elapsed = time.time() - t0
            print(f"\r  [{processed}] depth={depth} queue={len(queue)} "
                  f"book={len(book_entries)} ({elapsed:.0f}s)",
                  end="", file=sys.stderr, flush=True)

            engine.new_game()
            move_visits = expand_position(engine, pos_cmd, args.movetime)
            if not move_visits:
                continue

            total_visits = sum(v for _, v in move_visits)
            min_visits = int(total_visits * args.min_visit_ratio)

            filtered = [(m, v) for m, v in move_visits if v >= max(1, min_visits)]
            if not filtered:
                filtered = [move_visits[0]]

            book_entries[pos_cmd] = [(m, v) for m, v in filtered]

            top_moves = filtered[:args.branches]
            for move_str, _ in top_moves:
                child_moves = moves_so_far + [move_str]
                child_cmd = "position startpos moves " + " ".join(child_moves)
                queue.append((child_cmd, child_moves))

    except KeyboardInterrupt:
        print("\n  Interrupted, saving partial book...", file=sys.stderr)
    finally:
        engine.quit()

    elapsed = time.time() - t0
    print(f"\n  Processed {processed} positions in {elapsed:.0f}s", file=sys.stderr)

    # Write book file
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        if not args.append:
            f.write(f"# Alpha opening book (sims={args.simulations}, "
                    f"plies={args.plies}, branches={args.branches})\n")
            f.write(f"# {processed} positions analyzed\n\n")

        for pos_cmd in sorted(book_entries.keys(), key=lambda k: k.count(" ")):
            entries = book_entries[pos_cmd]
            f.write(f"{pos_cmd}\n")
            f.write(" ".join(f"{m}:{v}" for m, v in entries) + "\n\n")

    print(f"Wrote {len(book_entries)} positions to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
