#!/usr/bin/env python3
"""Versioned training loop with self-play evaluation.

Features:
  - Versioned model saves (models/nn_model_v001.pt, v002, ...)
  - Self-play against previous 1-2 versions every round
  - Time limit (default 3 hours), safe Ctrl-C interrupt
  - Auto-resume from last saved version
  - Progress and results logged to models/training_log.txt

Usage:
  python tools/train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe

  # Quick test (2 rounds, small scale)
  python tools/train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe \
      --rounds 2 --games-per-round 50 --epochs 2 --eval-games 4 --time-limit 600

  # Full 3-hour run
  python tools/train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe \
      --time-limit 10800

  # Resume after Ctrl-C (just run the same command again)
  python tools/train_loop.py --kifu kifu/floodgate --engine build/Release/kishi-to.exe
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


_interrupted = False


def _on_sigint(sig, frame):
    global _interrupted
    if _interrupted:
        print("\n[!] Force quit", file=sys.stderr, flush=True)
        sys.exit(1)
    _interrupted = True
    print("\n[!] Ctrl-C received. Finishing current step, then stopping...",
          file=sys.stderr, flush=True)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def find_existing_versions(models_dir: Path) -> list[int]:
    versions = []
    for f in models_dir.glob("nn_model_v*.pt"):
        try:
            v = int(f.stem.replace("nn_model_v", ""))
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(result.stderr)

    if result.returncode not in (0, 1):
        log(f"Self-play failed with code {result.returncode}")
        return None

    summary = result.stdout.strip()
    try:
        pct_str = [p for p in summary.split() if p.endswith("%)")]
        win_rate = float(pct_str[0].strip("(%)")) if pct_str else 50.0
    except (IndexError, ValueError):
        win_rate = 50.0

    return {"summary": summary, "win_rate": win_rate}


def append_log(log_path: Path, line: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Versioned training loop with self-play evaluation")
    parser.add_argument("--kifu", required=True, help="Kifu directory")
    parser.add_argument("--engine", required=True, help="USI engine executable")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Max training rounds (default: 100)")
    parser.add_argument("--time-limit", type=int, default=10800,
                        help="Time limit in seconds (default: 10800 = 3 hours)")
    parser.add_argument("--games-per-round", type=int, default=500,
                        help="Kifu games per round (cumulative)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per round")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Self-play games per evaluation")
    parser.add_argument("--movetime", type=int, default=1000,
                        help="Time per move in self-play (ms)")
    parser.add_argument("--models-dir", default="models",
                        help="Model output directory")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter (with torch)")
    parser.add_argument("--nn-python", default=None,
                        help="Python path for engine's NNPython option")
    parser.add_argument("--min-rate", type=int, default=0,
                        help="Minimum player rating filter")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="Middle-game sampling rate")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = models_dir / "training_log.txt"
    latest_model = models_dir / "nn_model.pt"

    existing = find_existing_versions(models_dir)
    start_version = max(existing) + 1 if existing else 1

    session_start = time.time()
    deadline = session_start + args.time_limit

    print(file=sys.stderr)
    log("=" * 56)
    log("  KishiTo Training Loop")
    log("=" * 56)
    log(f"  Kifu:        {args.kifu}")
    log(f"  Engine:      {args.engine}")
    log(f"  Device:      {args.device}")
    log(f"  Games/round: {args.games_per_round}")
    log(f"  Epochs:      {args.epochs}")
    log(f"  Eval games:  {args.eval_games}")
    log(f"  Time limit:  {fmt_duration(args.time_limit)}")
    if existing:
        log(f"  Resuming:    from v{start_version:03d} "
            f"(found v{existing[0]:03d}..v{existing[-1]:03d})")
    else:
        log(f"  Starting:    fresh (v001)")
    log("=" * 56)
    print(file=sys.stderr)

    append_log(log_path,
               f"\n--- Session {datetime.now().strftime('%Y-%m-%d %H:%M')} "
               f"(limit {fmt_duration(args.time_limit)}) ---")

    rounds_done = 0

    for round_idx in range(args.rounds):
        if _interrupted:
            log("Interrupted before starting next round.")
            break

        remaining = deadline - time.time()
        if remaining <= 0:
            log(f"Time limit reached ({fmt_duration(args.time_limit)}).")
            break

        version = start_version + round_idx
        total_games = args.games_per_round * version
        version_path = models_dir / f"nn_model_v{version:03d}.pt"
        elapsed_total = time.time() - session_start

        print(file=sys.stderr)
        log(f"{'='*56}")
        log(f"  Round {round_idx + 1} | v{version:03d} | "
            f"elapsed {fmt_duration(elapsed_total)} | "
            f"remaining {fmt_duration(remaining)}")
        log(f"{'='*56}")

        # --- Train ---
        log(f"Training: {total_games} games, {args.epochs} epochs...")
        t0 = time.time()
        resume = latest_model.exists()

        ok = run_training(
            args.python, args.kifu, str(latest_model), args.device,
            total_games, args.epochs, resume,
            args.min_rate, args.sample_rate,
        )
        train_elapsed = time.time() - t0

        if _interrupted:
            log("Interrupted during training.")
            break

        if not ok:
            msg = f"v{version:03d}: Training FAILED ({fmt_duration(train_elapsed)})"
            log(msg)
            append_log(log_path, msg)
            continue

        shutil.copy2(str(latest_model), str(version_path))
        log(f"Saved {version_path.name} ({fmt_duration(train_elapsed)})")
        append_log(log_path,
                   f"v{version:03d}: {total_games} games, {args.epochs} epochs, "
                   f"{fmt_duration(train_elapsed)}")

        # --- Self-play vs N-1 ---
        if version >= 2 and not _interrupted:
            prev_path = models_dir / f"nn_model_v{version - 1:03d}.pt"
            if prev_path.exists():
                log(f"Self-play: v{version:03d} vs v{version - 1:03d} "
                    f"({args.eval_games} games)...")
                result = run_eval(
                    args.python, args.engine,
                    str(version_path), str(prev_path),
                    args.eval_games, args.movetime, args.device,
                    args.nn_python,
                )
                if result:
                    mark = "+" if result["win_rate"] >= 50.0 else "-"
                    log(f"  [{mark}] {result['summary']}")
                    append_log(log_path,
                               f"  vs v{version - 1:03d}: {result['summary']}")

        # --- Self-play vs N-2 ---
        if version >= 3 and not _interrupted:
            prev2_path = models_dir / f"nn_model_v{version - 2:03d}.pt"
            if prev2_path.exists():
                log(f"Self-play: v{version:03d} vs v{version - 2:03d} "
                    f"({args.eval_games} games)...")
                result = run_eval(
                    args.python, args.engine,
                    str(version_path), str(prev2_path),
                    args.eval_games, args.movetime, args.device,
                    args.nn_python,
                )
                if result:
                    mark = "+" if result["win_rate"] >= 50.0 else "-"
                    log(f"  [{mark}] {result['summary']}")
                    append_log(log_path,
                               f"  vs v{version - 2:03d}: {result['summary']}")

        rounds_done += 1

    # --- Summary ---
    total_elapsed = time.time() - session_start
    final_versions = find_existing_versions(models_dir)
    print(file=sys.stderr)
    log("=" * 56)
    log("  Training Complete")
    log("=" * 56)
    log(f"  Rounds done: {rounds_done}")
    log(f"  Total time:  {fmt_duration(total_elapsed)}")
    if final_versions:
        log(f"  Versions:    v{final_versions[0]:03d} .. v{final_versions[-1]:03d}")
    log(f"  Latest:      {latest_model}")
    log(f"  Log:         {log_path}")
    if _interrupted:
        log(f"  (interrupted — run again to resume from v{final_versions[-1]+1:03d})"
            if final_versions else "  (interrupted)")
    log("=" * 56)

    append_log(log_path,
               f"--- Session ended {datetime.now().strftime('%Y-%m-%d %H:%M')} "
               f"({rounds_done} rounds, {fmt_duration(total_elapsed)}) ---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
