#!/usr/bin/env python3
"""Train the GPU engine's MLP evaluation from Floodgate kifu.

Reuses train_classic.py for CSA parsing, then calls kishi-to-classic
--extract-features to produce feature vectors, and finally trains the
MLP model via gpu_eval.py.

Usage:
  python tools/train_gpu.py --kifu kifu/floodgate --engine build/kishi-to-classic
  python tools/train_gpu.py --kifu kifu/floodgate --engine build/kishi-to-classic --epochs 5 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from train_classic import parse_game_for_classic

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def _parse_worker(args_tuple):
    filepath, min_rate, skip_opening, sample_rate = args_tuple
    return parse_game_for_classic(Path(filepath), min_rate=min_rate,
                                  skip_opening=skip_opening,
                                  sample_rate=sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Train GPU engine MLP from Floodgate kifu")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--engine", required=True,
                        help="kishi-to-classic executable (for feature extraction)")
    parser.add_argument("--model", default="gpu_model.pt",
                        help="Output model path (default: gpu_model.pt)")
    parser.add_argument("--min-rate", type=int, default=1500,
                        help="Minimum player rating (default: 1500)")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Max games to process (0=all)")
    parser.add_argument("--skip-opening", type=int, default=10,
                        help="Skip first N moves (default: 10)")
    parser.add_argument("--sample-rate", type=float, default=0.5,
                        help="Position sampling rate (default: 0.5)")
    parser.add_argument("--negatives", type=int, default=1,
                        help="Negative samples per position (default: 1)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--device", default="auto",
                        help="PyTorch device (default: auto)")
    args = parser.parse_args()

    kifu_dir = Path(args.kifu)
    csa_files = sorted(kifu_dir.rglob("*.csa"))
    print(f"Found {len(csa_files)} CSA files", file=sys.stderr)

    if args.max_games > 0:
        csa_files = csa_files[:args.max_games]

    # Step 1: Parse CSA -> SFEN+move TSV (same as train_classic.py)
    workers = max(1, os.cpu_count() or 1)
    print(f"Step 1: Parsing kifu ({workers} workers)...", file=sys.stderr)
    all_samples = []
    games_ok = 0
    skipped = 0

    work_args = [(str(f), args.min_rate, args.skip_opening, args.sample_rate)
                 for f in csa_files]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for samples in tqdm(executor.map(_parse_worker, work_args, chunksize=16),
                            total=len(csa_files), desc="Parsing kifu",
                            unit="file", file=sys.stderr):
            if samples is None:
                skipped += 1
                continue
            all_samples.extend(samples)
            games_ok += 1

    print(f"  Parsed {games_ok} games, {len(all_samples)} positions "
          f"(skipped {skipped})", file=sys.stderr)

    if not all_samples:
        print("No training data. Check kifu path and filters.", file=sys.stderr)
        return 1

    random.shuffle(all_samples)

    # Write intermediate SFEN+move TSV
    model_dir = Path(args.model).parent
    if str(model_dir) and model_dir != Path(""):
        model_dir.mkdir(parents=True, exist_ok=True)
    positions_file = model_dir / "gpu_positions.tsv"
    features_file = model_dir / "gpu_training.tsv"

    with open(positions_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(f"{s['sfen']}\t{s['usi_move']}\n")
    print(f"  Wrote {len(all_samples)} positions to {positions_file}",
          file=sys.stderr)

    # Step 2: Extract features via C++ engine
    print("Step 2: Extracting features...", file=sys.stderr)
    engine_path = str(Path(args.engine).resolve())
    extract_cmd = [
        engine_path,
        "--extract-features", str(positions_file.resolve()),
        "--output", str(features_file.resolve()),
        "--negatives", str(args.negatives),
    ]
    print(f"  Running: {' '.join(extract_cmd)}", file=sys.stderr)
    result = subprocess.run(extract_cmd, stderr=sys.stderr)
    if result.returncode != 0:
        print("Feature extraction failed.", file=sys.stderr)
        return 1

    # Step 3: Train MLP
    print("Step 3: Training MLP...", file=sys.stderr)
    tools_dir = Path(__file__).parent
    train_cmd = [
        sys.executable, str(tools_dir / "gpu_eval.py"), "train",
        "--data", str(features_file.resolve()),
        "--model", str(Path(args.model).resolve()),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    print(f"  Running: {' '.join(train_cmd)}", file=sys.stderr)
    result = subprocess.run(train_cmd, stderr=sys.stderr)
    if result.returncode != 0:
        print("Training failed.", file=sys.stderr)
        return 1

    print(f"Done. Model saved to {args.model}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
