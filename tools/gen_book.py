#!/usr/bin/env python3
"""Generate opening book from Floodgate CSA game records.

Parses only the header and first N moves of each file for speed.

Usage:
  python tools/gen_book.py --input kifu/floodgate --output build/Release/book.txt
  python tools/gen_book.py --input kifu/floodgate --min-rate 2800 --plies 8
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from csa_parser import (
    Board, CSA_PIECE_MAP, PIECE_TYPE_FROM_ID,
    idx, is_hirate, parse_csa_initial_board,
)


def parse_opening(filepath: Path, max_plies: int, min_rate: int
                  ) -> Optional[tuple[list[str], bool]]:
    """Parse only header + first max_plies moves from a CSA file.

    Returns (usi_moves, black_win) or None if skipped.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    lines = [l.rstrip("\n\r") for l in lines]

    black_rate = 0
    white_rate = 0
    black_win: Optional[bool] = None
    board_lines: list[str] = []
    move_lines: list[str] = []

    for line in lines:
        if line.startswith("'black_rate:"):
            m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
            if m:
                black_rate = int(float(m.group(1)))
        elif line.startswith("'white_rate:"):
            m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
            if m:
                white_rate = int(float(m.group(1)))
        elif line.startswith("'summary:"):
            parts = line.split(":")
            if len(parts) >= 4:
                p1 = parts[2].split()[-1] if parts[2] else ""
                p2 = parts[3].split()[-1] if len(parts) > 3 and parts[3] else ""
                if p1 == "win":
                    black_win = True
                elif p2 == "win":
                    black_win = False
                elif p1 == "lose":
                    black_win = False
                elif p2 == "lose":
                    black_win = True
        elif re.match(r"^P[1-9+\-]", line):
            board_lines.append(line)
        elif len(move_lines) < max_plies and len(line) >= 7 and \
                line[0] in ("+", "-") and line[1].isdigit():
            move_lines.append(line)

    if black_win is None:
        return None
    if min_rate > 0 and (black_rate < min_rate or white_rate < min_rate):
        return None
    if not is_hirate(board_lines):
        return None

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

    usi_moves: list[str] = []
    for mline in move_lines:
        side = 1 if mline[0] == "+" else -1
        from_file = int(mline[1])
        from_rank = int(mline[2])
        to_file = int(mline[3])
        to_rank = int(mline[4])
        piece_name = mline[5:7]
        pt = CSA_PIECE_MAP.get(piece_name, 0)
        if pt == 0:
            break

        is_drop = from_file == 0 and from_rank == 0
        to_sq = idx(to_file, to_rank)

        if is_drop:
            usi = Board.move_to_usi(-1, to_sq, True, pt, False)
            board.apply_move(-1, to_sq, True, pt, False, side)
        else:
            from_sq = idx(from_file, from_rank)
            current = abs(board.squares[from_sq]) if from_sq >= 0 else 0
            promote = pt >= 9 and current < 9 if current > 0 else False
            usi = Board.move_to_usi(from_sq, to_sq, False, 0, promote)
            board.apply_move(from_sq, to_sq, False, pt, promote, side)

        usi_moves.append(usi)

    return usi_moves, black_win


class MoveStats:
    __slots__ = ("count", "wins")

    def __init__(self):
        self.count = 0
        self.wins = 0.0


def generate_book(input_dir: str, output: str, min_rate: int, max_plies: int,
                  min_count: int):
    csa_files = sorted(Path(input_dir).rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    positions: dict[str, dict[str, MoveStats]] = defaultdict(lambda: defaultdict(MoveStats))
    games_used = 0
    skipped = 0

    for i, fp in enumerate(csa_files):
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(csa_files)} files, {games_used} games used...",
                  file=sys.stderr)

        result = parse_opening(fp, max_plies, min_rate)
        if result is None:
            skipped += 1
            continue

        usi_moves, black_win = result
        games_used += 1

        for ply in range(len(usi_moves)):
            if ply == 0:
                pos_key = "position startpos"
            else:
                pos_key = "position startpos moves " + " ".join(usi_moves[:ply])

            move = usi_moves[ply]
            side_at_ply = 1 if ply % 2 == 0 else -1
            win_for_side = (side_at_ply == 1 and black_win) or \
                           (side_at_ply == -1 and not black_win)

            stats = positions[pos_key][move]
            stats.count += 1
            if win_for_side:
                stats.wins += 1.0

    total_positions = 0
    with open(output, "w", encoding="utf-8") as f:
        f.write(f"# Opening book from Floodgate (min_rate={min_rate}, plies={max_plies})\n")
        f.write(f"# {games_used} games used, {skipped} skipped\n\n")

        for pos_key in sorted(positions.keys(), key=lambda k: k.count(" ")):
            move_stats = positions[pos_key]
            entries = []
            for move, stats in move_stats.items():
                if stats.count < min_count:
                    continue
                win_rate = stats.wins / stats.count if stats.count > 0 else 0.5
                weight = max(1, int(stats.count * win_rate * 2))
                entries.append((move, weight, stats.count))

            if not entries:
                continue

            entries.sort(key=lambda x: -x[1])
            f.write(f"{pos_key}\n")
            f.write(" ".join(f"{m}:{w}" for m, w, _ in entries) + "\n\n")
            total_positions += 1

    print(f"Wrote {total_positions} positions to {output}", file=sys.stderr)
    print(f"({games_used} games, {skipped} skipped)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate opening book from Floodgate CSA game records")
    parser.add_argument("--input", required=True,
                        help="Directory containing CSA files")
    parser.add_argument("--output", default="book.txt",
                        help="Output book file path")
    parser.add_argument("--min-rate", type=int, default=2800,
                        help="Minimum rating for both players (default: 2800)")
    parser.add_argument("--plies", type=int, default=8,
                        help="Max opening plies to extract (default: 8)")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Minimum game count per move (default: 2)")
    args = parser.parse_args()

    generate_book(args.input, args.output, args.min_rate, args.plies,
                  args.min_count)


if __name__ == "__main__":
    main()
