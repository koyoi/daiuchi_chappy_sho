#!/usr/bin/env python3
"""AlphaZero-style reinforcement learning loop orchestrator.

Cycle:
  1. Self-play: generate games with current best model
  2. Train: update model from replay buffer
  3. Evaluate: new checkpoint vs current best
  4. Gate: promote if win rate >= 55%

Usage:
  python alpha_train_loop.py --engine build/Release/kishi-to-alpha.exe --iterations 100
  python alpha_train_loop.py --engine build/Release/kishi-to-alpha.exe --resume --start-iter 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


GATE_THRESHOLD = 0.55
SELFPLAY_GAMES = 200
SELFPLAY_SIMULATIONS = 400
SELFPLAY_MOVETIME = 500
EVAL_GAMES = 20
EVAL_SIMULATIONS = 800
EVAL_MOVETIME = 1000
TRAIN_EPOCHS = 5
TRAIN_BATCH = 1024
TRAIN_LR = 1e-4
BUFFER_SIZE = 2000000
MAX_DATA_FILES = 30


def run_cmd(args: list[str], desc: str, cwd: str | None = None) -> int:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {desc}", file=sys.stderr)
    print(f"  cmd: {' '.join(args)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    t0 = time.time()
    result = subprocess.run(args, cwd=cwd)
    elapsed = time.time() - t0
    print(f"  -> exit={result.returncode} ({elapsed:.0f}s)", file=sys.stderr)
    return result.returncode


def evaluate_models(engine_path: str, model_new: str, model_best: str,
                    games: int, simulations: int, movetime: int,
                    device: str) -> float:
    """Play new model vs best model, return win rate for new model."""
    py = sys.executable
    selfplay_script = str(Path(__file__).parent / "alpha_self_play.py")

    wins_new = 0
    losses_new = 0
    draws = 0

    for game_idx in range(games):
        new_is_black = (game_idx % 2 == 0)

        black_model = model_new if new_is_black else model_best
        white_model = model_best if new_is_black else model_new

        args_black = [
            engine_path, "--model", black_model,
        ]
        args_white = [
            engine_path, "--model", white_model,
        ]

        result = play_eval_game(engine_path, black_model, white_model,
                                simulations, movetime, device)

        if result == "draw":
            draws += 1
        elif (result == "black" and new_is_black) or \
             (result == "white" and not new_is_black):
            wins_new += 1
        else:
            losses_new += 1

        total = wins_new + losses_new + draws
        wr = (wins_new + draws * 0.5) / total if total > 0 else 0.5
        print(f"  eval {game_idx+1}/{games}: W={wins_new} L={losses_new} D={draws} "
              f"WR={wr:.1%}", file=sys.stderr)

    total = wins_new + losses_new + draws
    return (wins_new + draws * 0.5) / total if total > 0 else 0.5


def play_eval_game(engine_path: str, black_model: str, white_model: str,
                   simulations: int, movetime: int, device: str) -> str:
    """Play one evaluation game between two models. Returns 'black', 'white', or 'draw'."""
    from alpha_self_play import AlphaUSIEngine, detect_repetition, usi_to_board_move, hirate_board

    MAX_MOVES = 256

    engine_b = AlphaUSIEngine(engine_path, model=black_model, device=device,
                               simulations=simulations, temperature_drop=0)
    engine_w = AlphaUSIEngine(engine_path, model=white_model, device=device,
                               simulations=simulations, temperature_drop=0)
    engine_b.start()
    engine_w.start()

    board = hirate_board()
    moves: list[str] = []
    result = "draw"

    try:
        for ply in range(MAX_MOVES):
            if moves:
                pos_cmd = "position startpos moves " + " ".join(moves)
            else:
                pos_cmd = "position startpos"

            engine = engine_b if ply % 2 == 0 else engine_w
            bestmove, _ = engine.go(pos_cmd, movetime)

            if bestmove == "resign" or bestmove == "none":
                result = "white" if ply % 2 == 0 else "black"
                break

            usi_to_board_move(board, bestmove)
            moves.append(bestmove)

            if detect_repetition(moves):
                result = "draw"
                break
    finally:
        engine_b.quit()
        engine_w.quit()

    return result


def main():
    parser = argparse.ArgumentParser(description="AlphaZero RL training loop")
    parser.add_argument("--engine", required=True, help="Path to kishi-to-alpha.exe")
    parser.add_argument("--work-dir", default="alpha_rl_work", help="Working directory")
    parser.add_argument("--iterations", type=int, default=100, help="Number of RL iterations")
    parser.add_argument("--start-iter", type=int, default=0, help="Starting iteration (for resume)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing work dir")
    parser.add_argument("--device", default="auto", help="NN device")
    parser.add_argument("--selfplay-games", type=int, default=SELFPLAY_GAMES)
    parser.add_argument("--selfplay-sims", type=int, default=SELFPLAY_SIMULATIONS)
    parser.add_argument("--selfplay-movetime", type=int, default=SELFPLAY_MOVETIME)
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    parser.add_argument("--eval-sims", type=int, default=EVAL_SIMULATIONS)
    parser.add_argument("--eval-movetime", type=int, default=EVAL_MOVETIME)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (always promote)")
    args = parser.parse_args()

    work = Path(args.work_dir)
    data_dir = work / "data"
    models_dir = work / "models"
    log_file = work / "train_log.jsonl"

    for d in [work, data_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best_model_pt = models_dir / "best.pt"
    best_model_onnx = models_dir / "best.onnx"
    py = sys.executable
    tools_dir = str(Path(__file__).parent)

    if not args.resume and args.start_iter == 0:
        if not best_model_pt.exists():
            print("No best.pt found. Starting from scratch (random init).", file=sys.stderr)

    for iteration in range(args.start_iter, args.iterations):
        iter_start = time.time()
        print(f"\n{'#'*60}", file=sys.stderr)
        print(f"  ITERATION {iteration + 1}/{args.iterations}", file=sys.stderr)
        print(f"{'#'*60}", file=sys.stderr)

        # --- Step 1: Self-play ---
        sp_output = data_dir / f"selfplay_{iteration:04d}.npz"
        sp_model = str(best_model_onnx) if best_model_onnx.exists() else ""

        sp_args = [
            py, os.path.join(tools_dir, "alpha_self_play.py"),
            "--engine", args.engine,
            "--games", str(args.selfplay_games),
            "--simulations", str(args.selfplay_sims),
            "--movetime", str(args.selfplay_movetime),
            "--output", str(sp_output),
        ]
        if sp_model:
            sp_args += ["--model", sp_model]
        if args.device != "auto":
            sp_args += ["--device", args.device]

        rc = run_cmd(sp_args, f"Self-play: {args.selfplay_games} games @ {args.selfplay_sims} sims")
        if rc != 0:
            print(f"Self-play failed (exit {rc}), skipping iteration", file=sys.stderr)
            continue

        # --- Step 2: Train ---
        data_files = sorted(data_dir.glob("selfplay_*.npz"))
        if len(data_files) > MAX_DATA_FILES:
            data_files = data_files[-MAX_DATA_FILES:]

        candidate_pt = models_dir / f"candidate_{iteration:04d}.pt"
        candidate_onnx = models_dir / f"candidate_{iteration:04d}.onnx"

        train_args = [
            py, os.path.join(tools_dir, "train_alpha_rl.py"),
            "--data", *[str(f) for f in data_files],
            "--model", str(best_model_pt) if best_model_pt.exists() else str(candidate_pt),
            "--output", str(candidate_onnx),
            "--epochs", str(TRAIN_EPOCHS),
            "--batch-size", str(TRAIN_BATCH),
            "--lr", str(TRAIN_LR),
            "--buffer-size", str(BUFFER_SIZE),
            "--channels", str(args.channels),
            "--blocks", str(args.blocks),
        ]

        rc = run_cmd(train_args, f"Training on {len(data_files)} data files")
        if rc != 0:
            print(f"Training failed (exit {rc}), skipping iteration", file=sys.stderr)
            continue

        saved_pt = best_model_pt if best_model_pt.exists() else candidate_pt
        if saved_pt != candidate_pt:
            shutil.copy2(saved_pt, candidate_pt)
        shutil.copy2(str(candidate_onnx), str(candidate_onnx))

        # --- Step 3: Evaluate ---
        promoted = False
        if args.skip_eval or not best_model_onnx.exists():
            promoted = True
            win_rate = 1.0
            print("  Promoted (no baseline or skip-eval)", file=sys.stderr)
        else:
            print(f"\n  Evaluating: candidate vs best ({args.eval_games} games)", file=sys.stderr)
            win_rate = evaluate_models(
                args.engine, str(candidate_onnx), str(best_model_onnx),
                args.eval_games, args.eval_sims, args.eval_movetime,
                args.device,
            )
            promoted = win_rate >= GATE_THRESHOLD
            status = "PROMOTED" if promoted else "REJECTED"
            print(f"  Win rate: {win_rate:.1%} -> {status} (threshold: {GATE_THRESHOLD:.0%})",
                  file=sys.stderr)

        # --- Step 4: Gate ---
        if promoted:
            shutil.copy2(str(candidate_pt), str(best_model_pt))
            shutil.copy2(str(candidate_onnx), str(best_model_onnx))
            print(f"  Best model updated: iteration {iteration + 1}", file=sys.stderr)

        iter_elapsed = time.time() - iter_start
        log_entry = {
            "iteration": iteration + 1,
            "win_rate": round(win_rate, 4),
            "promoted": promoted,
            "data_files": len(data_files),
            "elapsed_sec": round(iter_elapsed),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"\n  Iteration {iteration + 1} complete ({iter_elapsed:.0f}s). "
              f"WR={win_rate:.1%} {'PROMOTED' if promoted else 'rejected'}",
              file=sys.stderr)

        # Cleanup old candidates
        for old in models_dir.glob("candidate_*.pt"):
            if old != candidate_pt:
                old.unlink(missing_ok=True)
        for old in models_dir.glob("candidate_*.onnx"):
            if old != candidate_onnx:
                old.unlink(missing_ok=True)

    print("\nRL loop complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
