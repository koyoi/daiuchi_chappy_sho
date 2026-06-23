#!/usr/bin/env python3
"""Build a lightweight metadata index from Floodgate CSA files.

Reads only the header of each file (no move parsing) to extract player
names, ratings, result, and date.  Writes a TSV index and prints a
year-by-year summary so you can quickly decide which data to use for
training.

Usage:
  python tools/index_kifu.py --kifu kifu/floodgate
  python tools/index_kifu.py --kifu kifu/floodgate --output kifu/floodgate/index.tsv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def extract_meta(filepath: str) -> Optional[dict]:
    """Extract metadata from a CSA file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip("\n\r") for l in f]
    except Exception:
        return None

    black_name = ""
    white_name = ""
    black_rate = 0.0
    white_rate = 0.0
    result = ""
    start_time = ""
    num_moves = 0

    for line in lines:
        if line.startswith("N+"):
            black_name = line[2:]
        elif line.startswith("N-"):
            white_name = line[2:]
        elif line.startswith("'black_rate:"):
            m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
            if m:
                black_rate = float(m.group(1))
        elif line.startswith("'white_rate:"):
            m = re.search(r":(\d+(?:\.\d+)?)\s*$", line)
            if m:
                white_rate = float(m.group(1))
        elif line.startswith("'summary:"):
            result = line[len("'summary:"):]
        elif line.startswith("$START_TIME:"):
            start_time = line[len("$START_TIME:"):]

    if not start_time:
        m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", Path(filepath).name)
        if m:
            start_time = f"{m.group(1)}/{m.group(2)}/{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}"

    for line in lines:
        if len(line) >= 7 and line[0] in ('+', '-') and line[1].isdigit():
            num_moves += 1

    winner = ""
    if result:
        if "win" in result and "lose" in result:
            parts = result.split(":")
            if len(parts) >= 3:
                p1 = parts[1].strip().split()[-1] if parts[1] else ""
                p2 = parts[2].strip().split()[-1] if len(parts) > 2 and parts[2] else ""
                if p1 == "win":
                    winner = "black"
                elif p2 == "win":
                    winner = "white"
                elif p1 == "lose":
                    winner = "white"
                elif p2 == "lose":
                    winner = "black"
        elif "abnormal" in result:
            winner = "abnormal"
        elif "draw" in result:
            winner = "draw"

    year = ""
    if start_time:
        m = re.match(r"(\d{4})", start_time)
        if m:
            year = m.group(1)

    return {
        "file": filepath,
        "year": year,
        "date": start_time,
        "black": black_name,
        "white": white_name,
        "black_rate": black_rate,
        "white_rate": white_rate,
        "winner": winner,
        "moves": num_moves,
        "result_raw": result,
    }


def _worker(filepath: str) -> Optional[dict]:
    return extract_meta(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Build metadata index from Floodgate CSA files")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--output", default="", help="Output TSV path (default: <kifu>/index.tsv)")
    args = parser.parse_args()

    kifu_dir = Path(args.kifu)
    csa_files = sorted(str(f) for f in kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    if not csa_files:
        return 1

    workers = max(1, os.cpu_count() or 1)
    records = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for meta in tqdm(executor.map(_worker, csa_files, chunksize=64),
                         total=len(csa_files), desc="Indexing", unit="file",
                         file=sys.stderr):
            if meta is not None:
                records.append(meta)

    # Write index TSV
    output_path = args.output if args.output else str(kifu_dir / "index.tsv")
    cols = ["file", "year", "date", "black", "white", "black_rate", "white_rate",
            "winner", "moves", "result_raw"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in records:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {len(records)} records to {output_path}", file=sys.stderr)

    # Year-by-year summary
    by_year = defaultdict(lambda: {
        "total": 0, "black_win": 0, "white_win": 0, "draw": 0, "abnormal": 0,
        "rates": [], "moves": [],
    })
    for r in records:
        y = r["year"] or "unknown"
        s = by_year[y]
        s["total"] += 1
        if r["winner"] == "black":
            s["black_win"] += 1
        elif r["winner"] == "white":
            s["white_win"] += 1
        elif r["winner"] == "draw":
            s["draw"] += 1
        elif r["winner"] == "abnormal":
            s["abnormal"] += 1
        if r["black_rate"] > 0:
            s["rates"].append(r["black_rate"])
        if r["white_rate"] > 0:
            s["rates"].append(r["white_rate"])
        s["moves"].append(r["moves"])

    print(f"\n{'Year':>6}  {'Games':>7}  {'B win':>6}  {'W win':>6}  {'Draw':>5}  "
          f"{'Abn':>4}  {'AvgRate':>8}  {'MinRate':>8}  {'MaxRate':>8}  {'AvgMoves':>9}",
          file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for y in sorted(by_year.keys()):
        s = by_year[y]
        rates = s["rates"]
        moves = s["moves"]
        avg_rate = sum(rates) / len(rates) if rates else 0
        min_rate = min(rates) if rates else 0
        max_rate = max(rates) if rates else 0
        avg_moves = sum(moves) / len(moves) if moves else 0
        print(f"{y:>6}  {s['total']:>7}  {s['black_win']:>6}  {s['white_win']:>6}  "
              f"{s['draw']:>5}  {s['abnormal']:>4}  {avg_rate:>8.0f}  {min_rate:>8.0f}  "
              f"{max_rate:>8.0f}  {avg_moves:>9.1f}",
              file=sys.stderr)

    # Rating distribution
    all_rates = [r["black_rate"] for r in records if r["black_rate"] > 0] + \
                [r["white_rate"] for r in records if r["white_rate"] > 0]
    if all_rates:
        buckets = defaultdict(int)
        for rate in all_rates:
            bucket = int(rate // 500) * 500
            buckets[bucket] += 1
        print(f"\nRating distribution (player-appearances):", file=sys.stderr)
        for b in sorted(buckets.keys()):
            bar = "#" * (buckets[b] * 60 // max(buckets.values()))
            print(f"  {b:>5}-{b+499:<5}  {buckets[b]:>7}  {bar}", file=sys.stderr)

    # High-quality game count (both players >= threshold)
    for threshold in [1500, 2000, 2500, 3000]:
        count = sum(1 for r in records
                    if r["black_rate"] >= threshold and r["white_rate"] >= threshold
                    and r["winner"] in ("black", "white"))
        print(f"  Both >= {threshold}: {count} decisive games", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
