#!/usr/bin/env python3
"""Versioned training loop with self-play evaluation.

Each round:
  1. Train on kifu (cumulative, with --resume)
  2. Save versioned model
  3. Self-play against previous versions
  4. Log results

Usage:
  python train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe --rounds 10
  python train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe \
      --rounds 5 --games-per-round 500 --epochs 5 --eval-games 20
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def find_existing_versions(models_dir: Path) -> list[int]:
    versions = []
    for f in models_dir.glob("nn_model_v*.pt"):
        stem = f.stem
        try:
            v = int(stem.replace("nn_model_v", ""))
            versions.append(v)
        except ValueError:
            pass
    return sorted(versions)


def run_training(python: str, kifu: str, model_path: str, device: str,
                 max_games: int, epochs: int, resume: bool,
                 min_rate: int, sample_rate: float) -> bool:
    cmd = [
        python, str(Path(__file__).parent / "train.py"),
        "--kifu", kifu,
        "--model", model_path,
        "--device", device,
        "--max-games", str(max_games),
        "--epochs", str(epochs),
        "--min-rate", str(min_rate),
        "--sample-rate", str(sample_rate),
    ]
    if resume:
        cmd.append("--resume")

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stderr=sys.stderr)
    return result.returncode == 0


def run_eval(python: str, engine: str, model1: str, model2: str,
             games: int, movetime: int, device: str,
             nn_python: str | None = None) -> dict | None:
    cmd = [
        python, str(Path(__file__).parent / "self_play.py"),
        "--engine", engine,
        "--model1", model1,
        "--model2", model2,
        "--games", str(games),
        "--movetime", str(movetime),
    ]
    if nn_python:
        cmd.extend(["--python", nn_python])
    if device:
        cmd.extend(["--device", device])

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(result.stderr)

    if result.returncode not in (0, 1):
        log(f"Self-play failed with code {result.returncode}")
        return None

    summary = result.stdout.strip()
    parts = summary.split()
    try:
        pct_str = [p for p in parts if p.endswith("%)")]
        win_rate = float(pct_str[0].strip("(%)")) if pct_str else 50.0
    except (IndexError, ValueError):
        win_rate = 50.0

    return {"summary": summary, "win_rate": win_rate}


def append_log(log_path: Path, line: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Versioned training loop")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--engine", required=True, help="USI engine executable")
    parser.add_argument("--rounds", type=int, default=10, help="Training rounds")
    parser.add_argument("--games-per-round", type=int, default=500,
                        help="Kifu games per round (cumulative)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per round")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Self-play games per evaluation")
    parser.add_argument("--movetime", type=int, default=1000,
                        help="Time per move in self-play (ms)")
    parser.add_argument("--models-dir", default="models", help="Model output directory")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument("--nn-python", default=None,
                        help="Python path for engine NNPython option")
    parser.add_argument("--min-rate", type=int, default=0,
                        help="Minimum player rating filter")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="Middle-game sampling rate")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = models_dir / "training_log.txt"
    latest_model = models_dir / "nn_model.pt"

    existing = find_existing_versions(models_dir)
    start_version = max(existing) + 1 if existing else 1

    log(f"Starting from version {start_version}, {args.rounds} rounds")
    log(f"Kifu: {args.kifu}, {args.games_per_round} games/round, {args.epochs} epochs")
    append_log(log_path, f"\n--- Session started {datetime.now().isoformat()} ---")

    for round_idx in range(args.rounds):
        version = start_version + round_idx
        total_games = args.games_per_round * version
        version_path = models_dir / f"nn_model_v{version:03d}.pt"

        log(f"\n{'='*50}")
        log(f"=== Round {round_idx + 1}/{args.rounds} (v{version:03d}) ===")
        log(f"{'='*50}")
        log(f"Training with {total_games} games, {args.epochs} epochs...")

        t0 = time.time()
        resume = latest_model.exists()
        train_model = str(latest_model)

        ok = run_training(
            args.python, args.kifu, train_model, args.device,
            total_games, args.epochs, resume,
            args.min_rate, args.sample_rate,
        )
        elapsed = time.time() - t0

        if not ok:
            msg = f"v{version:03d}: Training FAILED after {elapsed:.0f}s"
            log(msg)
            append_log(log_path, msg)
            continue

        shutil.copy2(str(latest_model), str(version_path))
        log(f"Saved {version_path} ({elapsed:.0f}s)")
        append_log(log_path, f"v{version:03d}: trained {total_games} games, "
                             f"{args.epochs} epochs, {elapsed:.0f}s")

        if version >= 2:
            prev_path = models_dir / f"nn_model_v{version - 1:03d}.pt"
            if prev_path.exists():
                log(f"\nEvaluating v{version:03d} vs v{version - 1:03d} "
                    f"({args.eval_games} games)...")
                result = run_eval(
                    args.python, args.engine,
                    str(version_path), str(prev_path),
                    args.eval_games, args.movetime, args.device,
                    args.nn_python,
                )
                if result:
                    improved = result["win_rate"] >= 50.0
                    mark = "improved" if improved else "WORSE"
                    log(f"  {result['summary']} -> {mark}")
                    append_log(log_path,
                               f"  vs v{version - 1:03d}: {result['summary']} ({mark})")

        if version >= 3:
            prev2_path = models_dir / f"nn_model_v{version - 2:03d}.pt"
            if prev2_path.exists():
                log(f"\nEvaluating v{version:03d} vs v{version - 2:03d} "
                    f"({args.eval_games} games)...")
                result = run_eval(
                    args.python, args.engine,
                    str(version_path), str(prev2_path),
                    args.eval_games, args.movetime, args.device,
                    args.nn_python,
                )
                if result:
                    improved = result["win_rate"] >= 50.0
                    mark = "improved" if improved else "WORSE"
                    log(f"  {result['summary']} -> {mark}")
                    append_log(log_path,
                               f"  vs v{version - 2:03d}: {result['summary']} ({mark})")

    log(f"\n{'='*50}")
    log(f"All rounds complete. Latest model: {latest_model}")
    log(f"Training log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
