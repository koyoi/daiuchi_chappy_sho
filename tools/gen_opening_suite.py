#!/usr/bin/env python3
"""Generate opening position suite from Floodgate CSA game records.

Extracts opening move sequences (8-12 plies) for use in gate evaluation,
ensuring deterministic engines play diverse games.

Usage:
  python tools/gen_opening_suite.py --kifu kifu/floodgate --output opening_suite.txt
  python tools/gen_opening_suite.py --kifu kifu/floodgate --min-rate 2500 --max-positions 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from gen_book import parse_opening


def main():
    parser = argparse.ArgumentParser(description="Generate opening suite from CSA kifu")
    parser.add_argument("--kifu", required=True, help="Directory containing CSA files")
    parser.add_argument("--output", default="opening_suite.txt", help="Output file path")
    parser.add_argument("--min-rate", type=int, default=2500,
                        help="Minimum player rating (both players)")
    parser.add_argument("--min-plies", type=int, default=8,
                        help="Minimum opening length in plies")
    parser.add_argument("--max-plies", type=int, default=12,
                        help="Maximum opening length in plies")
    parser.add_argument("--max-positions", type=int, default=500,
                        help="Maximum number of positions to output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    kifu_dir = Path(args.kifu)
    if not kifu_dir.is_dir():
        print(f"Error: {kifu_dir} is not a directory", file=sys.stderr)
        return 1

    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    rng = random.Random(args.seed)
    openings: set[str] = set()
    parsed = 0
    skipped = 0

    for i, filepath in enumerate(csa_files):
        if i % 1000 == 0 and i > 0:
            print(f"  processed {i}/{len(csa_files)}, found {len(openings)} unique openings",
                  file=sys.stderr)

        result = parse_opening(filepath, args.max_plies, args.min_rate)
        if result is None:
            skipped += 1
            continue

        usi_moves, _ = result
        parsed += 1

        if len(usi_moves) < args.min_plies:
            continue
        n_plies = rng.randint(args.min_plies, min(args.max_plies, len(usi_moves)))

        opening = " ".join(usi_moves[:n_plies])
        openings.add(opening)

    print(f"Parsed {parsed} games, skipped {skipped}, "
          f"found {len(openings)} unique openings", file=sys.stderr)

    opening_list = sorted(openings)
    if len(opening_list) > args.max_positions:
        rng.shuffle(opening_list)
        opening_list = opening_list[:args.max_positions]
        opening_list.sort()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for opening in opening_list:
            f.write(opening + "\n")

    print(f"Wrote {len(opening_list)} openings to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
