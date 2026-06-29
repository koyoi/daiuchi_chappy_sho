#!/usr/bin/env python3
"""Alpha × NNUE cross-training loop.

Plays Alpha engine against NNUE engine, generating training data for both
from the same games. This avoids exploit convergence (ハメ技) that occurs
in pure self-play by exposing each engine to a fundamentally different
evaluation style.

Optional: MCTS engine as teacher, deep relabeling (Suishou-style gensfen).

Usage:
  python tools/cross_train_loop.py \
    --alpha-engine build/Release/kishi-to-alpha.exe \
    --nnue-engine build/Release/kishi-to-nnue.exe \
    --iterations 50

  # With MCTS teacher and deep relabeling
  python tools/cross_train_loop.py \
    --alpha-engine build/Release/kishi-to-alpha.exe \
    --nnue-engine build/Release/kishi-to-nnue.exe \
    --mcts-engine build/Release/kishi-to.exe \
    --relabel-movetime 2000 \
    --iterations 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

def _ensure_cuda_lib_path():
    """Add pip-installed NVIDIA CUDA 12 libraries to LD_LIBRARY_PATH."""
    dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.cudnn"):
        try:
            mod = __import__(pkg, fromlist=["lib"])
            if hasattr(mod, "__file__") and mod.__file__:
                lib_dir = str(Path(mod.__file__).parent / "lib")
            elif hasattr(mod, "__path__"):
                lib_dir = str(Path(list(mod.__path__)[0]) / "lib")
            else:
                continue
            if Path(lib_dir).is_dir():
                dirs.append(lib_dir)
        except ImportError:
            pass
    if dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        missing = [d for d in dirs if d not in existing]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(missing) + ((":" + existing) if existing else "")

_ensure_cuda_lib_path()

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha_self_play import (
    AlphaUSIEngine,
    detect_repetition,
    hirate_board,
    positions_to_arrays,
    usi_to_board_move,
    POLICY_SIZE,
)
from nnue_self_play import (
    USIEngine as NNUEUSIEngine,
    apply_usi_move,
    cp_to_label,
    save_data,
    make_startpos,
    BLACK_FIRST_MOVES,
    WHITE_FIRST_MOVES,
)
from train_nnue import extract_features_from_board

MAX_MOVES = 256
GATE_THRESHOLD = 0.55


def search_with_scores(engine: NNUEUSIEngine, pos_cmd: str, movetime: int
                       ) -> tuple[str | None, int | None, list[tuple[str, int]]]:
    """Search and then fetch root move scores via getscores command.

    Returns (bestmove, best_cp, [(move, score), ...]).
    """
    bestmove, cp = engine.search(pos_cmd, movetime)
    scores: list[tuple[str, int]] = []
    engine._send("getscores")
    lines = engine._wait_for("scores", timeout=5)
    for line in lines:
        if line.startswith("scores") and line != "scores none":
            for token in line.split()[1:]:
                parts = token.split(":")
                if len(parts) == 2:
                    try:
                        scores.append((parts[0], int(parts[1])))
                    except ValueError:
                        pass
            break
    return bestmove, cp, scores


def softmax_select(scores: list[tuple[str, int]], temperature: float) -> tuple[str, int]:
    """Select a move probabilistically based on scores and temperature.

    Returns (selected_move, selected_score).
    """
    if not scores:
        return "resign", 0
    if temperature <= 0 or len(scores) == 1:
        return scores[0]

    max_score = scores[0][1]
    weights = [np.exp((s - max_score) / temperature) for _, s in scores]
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = np.random.choice(len(scores), p=probs)
    return scores[idx]


def play_cross_game(
    alpha_engine: AlphaUSIEngine,
    nnue_engine: NNUEUSIEngine,
    alpha_is_black: bool,
    alpha_movetime: int,
    nnue_movetime: int,
    random_plies: int = 4,
    nnue_is_mcts: bool = False,
    nnue_softmax_temp: float = 0.0,
) -> tuple[list[dict], list[dict], str, int]:
    """Play one game between Alpha and NNUE, collecting training data for both.

    Returns (alpha_samples, nnue_samples, result, ply_count).
    result is 'black', 'white', or 'draw'.
    """
    alpha_engine.new_game()
    nnue_engine.new_game()

    board_alpha = hirate_board()
    board_nnue = make_startpos()
    moves: list[str] = []
    alpha_samples: list[dict] = []
    nnue_samples: list[dict] = []

    opening_moves = []
    if random_plies >= 1:
        opening_moves.append(random.choice(BLACK_FIRST_MOVES))
    if random_plies >= 2:
        opening_moves.append(random.choice(WHITE_FIRST_MOVES))

    result = "draw"

    for ply in range(MAX_MOVES):
        pos_cmd = "position startpos"
        if moves:
            pos_cmd += " moves " + " ".join(moves)

        is_alpha_turn = (ply % 2 == 0) == alpha_is_black

        if ply < len(opening_moves):
            bestmove = opening_moves[ply]
        elif ply < random_plies:
            if is_alpha_turn:
                bestmove, _ = alpha_engine.go(pos_cmd, 50)
            else:
                bestmove, _ = nnue_engine.search(pos_cmd, 50)
        elif is_alpha_turn:
            encoded = board_alpha.encode()
            side = board_alpha.side
            bestmove, visits = alpha_engine.go(pos_cmd, alpha_movetime)

            if bestmove not in ("resign", "none") and visits:
                alpha_samples.append({
                    "encoded": encoded,
                    "visits": visits,
                    "side": side,
                    "ply": ply,
                })
        else:
            bf, wf = extract_features_from_board(board_nnue)
            side_black = board_nnue.side == 1

            if nnue_softmax_temp > 0:
                bestmove, cp, scores = search_with_scores(
                    nnue_engine, pos_cmd, nnue_movetime)
                if scores and bestmove not in (None, "resign", "win"):
                    selected_move, selected_cp = softmax_select(scores, nnue_softmax_temp)
                    if selected_move != "resign":
                        bestmove = selected_move
            else:
                bestmove, cp = nnue_engine.search(pos_cmd, nnue_movetime)

            if bestmove not in (None, "resign", "win") and cp is not None and bf and wf:
                nnue_samples.append({
                    "black_feats": np.array(bf, dtype=np.int32),
                    "white_feats": np.array(wf, dtype=np.int32),
                    "side_black": side_black,
                    "soft_label": float(cp_to_label(cp, nnue_is_mcts)),
                    "ply": ply,
                    "pos_cmd": pos_cmd,
                })

        if bestmove in ("resign", "none", None):
            result = "white" if ply % 2 == 0 else "black"
            break

        usi_to_board_move(board_alpha, bestmove)
        apply_usi_move(board_nnue, bestmove)
        moves.append(bestmove)

        if detect_repetition(moves):
            result = "draw"
            break

    return alpha_samples, nnue_samples, result, len(moves)


def blend_nnue_labels_with_outcome(
    nnue_samples: list[dict],
    result: str,
    alpha_is_black: bool,
    total_plies: int,
    eval_weight: float = 0.7,
    halflife: float = 60.0,
):
    """Blend NNUE engine eval labels with game outcome (in-place)."""
    for sample in nnue_samples:
        nnue_is_black = sample["side_black"]
        if result == "draw":
            outcome = 0.5
        elif (result == "black" and nnue_is_black) or (result == "white" and not nnue_is_black):
            outcome = 1.0
        else:
            outcome = 0.0

        plies_remaining = total_plies - sample["ply"]
        discount = 0.5 ** (plies_remaining / halflife)
        outcome_label = 0.5 + discount * (outcome - 0.5)

        sample["soft_label"] = eval_weight * sample["soft_label"] + (1 - eval_weight) * outcome_label


def play_cross_games(
    alpha_engine: AlphaUSIEngine,
    nnue_engine: NNUEUSIEngine,
    num_games: int,
    alpha_movetime: int,
    nnue_movetime: int,
    random_plies: int = 4,
    nnue_is_mcts: bool = False,
    eval_weight: float = 0.7,
    nnue_softmax_temp: float = 0.0,
) -> tuple[list[list[dict]], list[str], list[dict]]:
    """Play multiple cross-games, returning Alpha and NNUE data.

    Returns (all_alpha_positions, all_alpha_results, all_nnue_samples).
    """
    all_alpha_positions: list[list[dict]] = []
    all_alpha_results: list[str] = []
    all_nnue_samples: list[dict] = []

    wins_alpha = 0
    wins_nnue = 0
    draws = 0

    for game_idx in range(num_games):
        alpha_is_black = (game_idx % 2 == 0)

        alpha_samples, nnue_samples, result, ply_count = play_cross_game(
            alpha_engine, nnue_engine, alpha_is_black,
            alpha_movetime, nnue_movetime, random_plies, nnue_is_mcts,
            nnue_softmax_temp,
        )

        blend_nnue_labels_with_outcome(
            nnue_samples, result, alpha_is_black, ply_count, eval_weight,
        )

        all_alpha_positions.append(alpha_samples)
        all_alpha_results.append(result)
        all_nnue_samples.extend(nnue_samples)

        alpha_color = "B" if alpha_is_black else "W"
        if result == "draw":
            draws += 1
        elif (result == "black") == alpha_is_black:
            wins_alpha += 1
        else:
            wins_nnue += 1

        print(
            f"  Game {game_idx+1}/{num_games}: Alpha({alpha_color}) "
            f"result={result} ({ply_count}mv) "
            f"[A:{wins_alpha} N:{wins_nnue} D:{draws} "
            f"alpha_pos:{sum(len(p) for p in all_alpha_positions)} "
            f"nnue_pos:{len(all_nnue_samples)}]",
            file=sys.stderr,
        )

    return all_alpha_positions, all_alpha_results, all_nnue_samples


def relabel_positions(
    engine: NNUEUSIEngine,
    nnue_samples: list[dict],
    movetime: int,
    is_mcts: bool = False,
):
    """Re-evaluate NNUE positions with deeper search for better labels (in-place)."""
    print(f"  Relabeling {len(nnue_samples)} positions @ {movetime}ms...", file=sys.stderr)
    relabeled = 0
    for i, sample in enumerate(nnue_samples):
        pos_cmd = sample.get("pos_cmd")
        if not pos_cmd:
            continue
        _, cp = engine.search(pos_cmd, movetime)
        if cp is not None:
            sample["soft_label"] = float(cp_to_label(cp, is_mcts))
            relabeled += 1
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(nnue_samples)} relabeled", file=sys.stderr)
    print(f"  Relabeled {relabeled}/{len(nnue_samples)} positions", file=sys.stderr)


def evaluate_nnue(
    engine_path: str,
    new_weights: str,
    best_weights: str,
    games: int,
    movetime: int,
    hash_mb: int = 64,
) -> float:
    """Play new NNUE vs old NNUE, return win rate for new."""
    from self_play import play_one_game as sp_play_one_game, USIEngine as SPUSIEngine

    e_new = SPUSIEngine(engine_path)
    e_best = SPUSIEngine(engine_path)

    e_new.start()
    e_best.start()
    e_new._send(f"setoption name NNUEFile value {new_weights}")
    e_best._send(f"setoption name NNUEFile value {best_weights}")
    e_new._send("isready")
    e_best._send("isready")
    while True:
        line = e_new._recv()
        if line and line.strip() == "readyok":
            break
    while True:
        line = e_best._recv()
        if line and line.strip() == "readyok":
            break

    wins_new = 0
    losses_new = 0
    draws = 0

    def restart_engines():
        nonlocal e_new, e_best
        try:
            e_new.quit()
        except Exception:
            pass
        try:
            e_best.quit()
        except Exception:
            pass
        e_new = SPUSIEngine(engine_path)
        e_best = SPUSIEngine(engine_path)
        e_new.start()
        e_best.start()
        e_new._send(f"setoption name NNUEFile value {new_weights}")
        e_best._send(f"setoption name NNUEFile value {best_weights}")
        e_new._send("isready")
        e_best._send("isready")
        while True:
            line = e_new._recv()
            if line and line.strip() == "readyok":
                break
        while True:
            line = e_best._recv()
            if line and line.strip() == "readyok":
                break

    try:
        for game_idx in range(games):
            new_is_black = (game_idx % 2 == 0)
            black_engine = e_new if new_is_black else e_best
            white_engine = e_best if new_is_black else e_new
            label = f"NNUE eval {game_idx+1}/{games}"

            try:
                result, ply = sp_play_one_game(black_engine, white_engine, movetime, label)
            except (OSError, BrokenPipeError):
                print(f"  nnue eval {game_idx+1}: engine crashed, restarting...", file=sys.stderr)
                restart_engines()
                draws += 1
                total = wins_new + losses_new + draws
                wr = (wins_new + draws * 0.5) / total if total > 0 else 0.5
                print(
                    f"  nnue eval {game_idx+1}/{games}: W={wins_new} L={losses_new} D={draws} "
                    f"WR={wr:.1%} (crash→draw)",
                    file=sys.stderr,
                )
                continue

            if "draw" in result:
                draws += 1
            elif (result == "black" and new_is_black) or (result == "white" and not new_is_black):
                wins_new += 1
            else:
                losses_new += 1

            total = wins_new + losses_new + draws
            wr = (wins_new + draws * 0.5) / total if total > 0 else 0.5
            print(
                f"  nnue eval {game_idx+1}/{games}: W={wins_new} L={losses_new} D={draws} "
                f"WR={wr:.1%}",
                file=sys.stderr,
            )
    finally:
        e_new.quit()
        e_best.quit()

    total = wins_new + losses_new + draws
    return (wins_new + draws * 0.5) / total if total > 0 else 0.5


def run_cmd(args: list[str], desc: str) -> int:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {desc}", file=sys.stderr)
    print(f"  cmd: {' '.join(args)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    t0 = time.time()
    result = subprocess.run(args)
    elapsed = time.time() - t0
    print(f"  -> exit={result.returncode} ({elapsed:.0f}s)", file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Alpha x NNUE cross-training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--alpha-engine", required=True, help="Path to kishi-to-alpha.exe")
    parser.add_argument("--nnue-engine", required=True, help="Path to kishi-to-nnue.exe")
    parser.add_argument("--mcts-engine", default="", help="Optional MCTS teacher engine path")
    parser.add_argument("--work-dir", default="cross_train_work", help="Working directory")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--start-iter", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    g = parser.add_argument_group("cross-play")
    g.add_argument("--cross-games", type=int, default=200, help="Games per iteration")
    g.add_argument("--alpha-movetime", type=int, default=500, help="Alpha ms/move")
    g.add_argument("--nnue-movetime", type=int, default=500, help="NNUE ms/move")
    g.add_argument("--alpha-simulations", type=int, default=400)
    g.add_argument("--alpha-batch-size", type=int, default=64)
    g.add_argument("--random-plies", type=int, default=4)
    g.add_argument("--eval-weight", type=float, default=0.7,
                   help="NNUE label blend: eval_weight*eval + (1-eval_weight)*outcome")
    g.add_argument("--nnue-softmax-temp", type=float, default=0.0,
                   help="Softmax temperature for NNUE move selection (0=deterministic, try 100-300)")

    g = parser.add_argument_group("teacher / relabeling")
    g.add_argument("--teacher-games", type=int, default=0, help="Games vs MCTS teacher per iter")
    g.add_argument("--relabel-movetime", type=int, default=0, help="Deep NNUE relabel ms (0=off)")

    g = parser.add_argument_group("training")
    g.add_argument("--alpha-train-epochs", type=int, default=5)
    g.add_argument("--nnue-train-epochs", type=int, default=10)
    g.add_argument("--alpha-buffer-size", type=int, default=2000000)
    g.add_argument("--alpha-channels", type=int, default=192)
    g.add_argument("--alpha-blocks", type=int, default=15)
    g.add_argument("--nnue-batch-size", type=int, default=8192)
    g.add_argument("--nnue-lr", type=float, default=4e-3)

    g = parser.add_argument_group("evaluation")
    g.add_argument("--eval-games", type=int, default=20)
    g.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD)
    g.add_argument("--eval-movetime", type=int, default=1000)
    g.add_argument("--alpha-eval-sims", type=int, default=800)
    g.add_argument("--skip-eval", action="store_true", help="Skip eval, always promote")

    g = parser.add_argument_group("models")
    g.add_argument("--alpha-model", default="", help="Initial Alpha model .onnx")
    g.add_argument("--nnue-weights", default="", help="Initial NNUE weights .bin")
    g.add_argument("--device", default="auto")

    args = parser.parse_args()

    work = Path(args.work_dir)
    alpha_data_dir = work / "alpha" / "data"
    alpha_models_dir = work / "alpha" / "models"
    nnue_data_dir = work / "nnue" / "data"
    nnue_models_dir = work / "nnue" / "models"
    log_file = work / "train_log.jsonl"

    for d in [alpha_data_dir, alpha_models_dir, nnue_data_dir, nnue_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    alpha_best_pt = alpha_models_dir / "best.pt"
    alpha_best_onnx = alpha_models_dir / "best.onnx"
    nnue_best_pt = nnue_models_dir / "best.pt"
    nnue_best_bin = nnue_models_dir / "best.bin"

    if args.alpha_model and not alpha_best_onnx.exists():
        shutil.copy2(args.alpha_model, alpha_best_onnx)
        print(f"Copied initial Alpha model to {alpha_best_onnx}", file=sys.stderr)
    if args.nnue_weights and not nnue_best_bin.exists():
        shutil.copy2(args.nnue_weights, nnue_best_bin)
        print(f"Copied initial NNUE weights to {nnue_best_bin}", file=sys.stderr)

    py = sys.executable
    tools_dir = str(Path(__file__).parent)

    for iteration in range(args.start_iter, args.iterations):
        iter_start = time.time()
        print(f"\n{'#'*60}", file=sys.stderr)
        print(f"  ITERATION {iteration + 1}/{args.iterations}", file=sys.stderr)
        print(f"{'#'*60}", file=sys.stderr)

        # ---- Phase 1: Cross-play (Alpha vs NNUE) ----
        print("\n--- Phase 1: Cross-play ---", file=sys.stderr)

        alpha_eng = AlphaUSIEngine(
            args.alpha_engine,
            model=str(alpha_best_onnx) if alpha_best_onnx.exists() else None,
            device=args.device if args.device != "auto" else None,
            simulations=args.alpha_simulations,
            batch_size=args.alpha_batch_size,
            temperature_drop=30,
        )
        nnue_extra = {}
        if nnue_best_bin.exists():
            nnue_extra["NNUEFile"] = str(nnue_best_bin)
        nnue_eng = NNUEUSIEngine(args.nnue_engine, extra_options=nnue_extra)

        alpha_eng.start()

        try:
            all_alpha_positions, all_alpha_results, all_nnue_samples = play_cross_games(
                alpha_eng, nnue_eng, args.cross_games,
                args.alpha_movetime, args.nnue_movetime,
                args.random_plies, nnue_is_mcts=False,
                eval_weight=args.eval_weight,
                nnue_softmax_temp=args.nnue_softmax_temp,
            )
        finally:
            alpha_eng.quit()
            nnue_eng.quit()

        # ---- Phase 2: Optional deep relabeling ----
        if args.relabel_movetime > 0 and all_nnue_samples:
            print("\n--- Phase 2: Deep relabeling ---", file=sys.stderr)
            relabel_eng = NNUEUSIEngine(
                args.nnue_engine, hash_mb=128,
                extra_options={"NNUEFile": str(nnue_best_bin)} if nnue_best_bin.exists() else None,
            )
            try:
                relabel_positions(relabel_eng, all_nnue_samples, args.relabel_movetime)
            finally:
                relabel_eng.quit()

        # ---- Phase 3: Optional teacher games (vs MCTS) ----
        if args.mcts_engine and args.teacher_games > 0:
            print("\n--- Phase 3: Teacher games (vs MCTS) ---", file=sys.stderr)

            mcts_eng = NNUEUSIEngine(args.mcts_engine)
            alpha_eng2 = AlphaUSIEngine(
                args.alpha_engine,
                model=str(alpha_best_onnx) if alpha_best_onnx.exists() else None,
                device=args.device if args.device != "auto" else None,
                simulations=args.alpha_simulations,
                batch_size=args.alpha_batch_size,
                temperature_drop=30,
            )
            alpha_eng2.start()

            try:
                t_alpha_pos, t_alpha_res, _ = play_cross_games(
                    alpha_eng2, mcts_eng, args.teacher_games,
                    args.alpha_movetime, args.nnue_movetime,
                    args.random_plies, nnue_is_mcts=True,
                    eval_weight=args.eval_weight,
                )
                all_alpha_positions.extend(t_alpha_pos)
                all_alpha_results.extend(t_alpha_res)
            finally:
                alpha_eng2.quit()
                mcts_eng.quit()

            # NNUE vs MCTS: MCTS plays the "alpha" role (we discard its
            # visit data), NNUE data is collected from the "nnue" role.
            mcts_as_alpha = AlphaUSIEngine(
                args.mcts_engine,
                simulations=args.alpha_simulations,
                temperature_drop=0,
            )
            mcts_as_alpha.start()
            nnue_eng_t = NNUEUSIEngine(
                args.nnue_engine,
                extra_options={"NNUEFile": str(nnue_best_bin)} if nnue_best_bin.exists() else None,
            )
            try:
                _, _, t_nnue_samples = play_cross_games(
                    mcts_as_alpha, nnue_eng_t, args.teacher_games,
                    args.nnue_movetime, args.nnue_movetime,
                    args.random_plies, nnue_is_mcts=False,
                    eval_weight=args.eval_weight,
                )
                all_nnue_samples.extend(t_nnue_samples)
            finally:
                mcts_as_alpha.quit()
                nnue_eng_t.quit()

        # ---- Phase 4: Save data ----
        print("\n--- Phase 4: Save data ---", file=sys.stderr)

        alpha_data_path = alpha_data_dir / f"cross_{iteration:04d}.npz"
        arrays = positions_to_arrays(all_alpha_positions, all_alpha_results)
        if arrays is not None:
            encoded, policy, wdl = arrays
            np.savez(str(alpha_data_path), encoded=encoded, policy_target=policy, wdl_target=wdl)
            print(f"  Alpha: {len(encoded)} positions -> {alpha_data_path}", file=sys.stderr)
        else:
            print("  Alpha: no positions collected", file=sys.stderr)

        nnue_data_path = nnue_data_dir / f"cross_{iteration:04d}.npz"
        if all_nnue_samples:
            meta = {
                "source": "cross_train",
                "teacher": "nnue_cross",
                "movetime": args.nnue_movetime,
                "games": args.cross_games,
                "random_plies": args.random_plies,
            }
            save_data(all_nnue_samples, str(nnue_data_path), meta)
            print(f"  NNUE: {len(all_nnue_samples)} positions -> {nnue_data_path}", file=sys.stderr)
        else:
            print("  NNUE: no positions collected", file=sys.stderr)

        # ---- Phase 5: Train Alpha ----
        alpha_data_files = sorted(alpha_data_dir.glob("cross_*.npz"))
        if len(alpha_data_files) > 30:
            alpha_data_files = alpha_data_files[-30:]

        if alpha_data_files:
            print("\n--- Phase 5: Train Alpha ---", file=sys.stderr)
            alpha_candidate_pt = alpha_models_dir / f"candidate_{iteration:04d}.pt"
            alpha_candidate_onnx = alpha_models_dir / f"candidate_{iteration:04d}.onnx"

            train_args = [
                py, os.path.join(tools_dir, "train_alpha_rl.py"),
                "--data", *[str(f) for f in alpha_data_files],
                "--model", str(alpha_best_pt) if alpha_best_pt.exists() else str(alpha_candidate_pt),
                "--output", str(alpha_candidate_onnx),
                "--epochs", str(args.alpha_train_epochs),
                "--batch-size", "1024",
                "--lr", "1e-4",
                "--buffer-size", str(args.alpha_buffer_size),
                "--channels", str(args.alpha_channels),
                "--blocks", str(args.alpha_blocks),
            ]
            rc = run_cmd(train_args, f"Alpha training ({len(alpha_data_files)} data files)")

            if rc == 0:
                # ---- Phase 7: Gate Alpha ----
                alpha_promoted = False
                if args.skip_eval or not alpha_best_onnx.exists():
                    alpha_promoted = True
                    alpha_wr = 1.0
                    print("  Alpha promoted (no baseline or skip-eval)", file=sys.stderr)
                else:
                    print(f"\n--- Gate Alpha ({args.eval_games} games) ---", file=sys.stderr)
                    from alpha_train_loop import evaluate_models
                    alpha_wr = evaluate_models(
                        args.alpha_engine, str(alpha_candidate_onnx), str(alpha_best_onnx),
                        args.eval_games, args.alpha_eval_sims, args.eval_movetime,
                        args.device,
                    )
                    alpha_promoted = alpha_wr >= args.gate_threshold
                    status = "PROMOTED" if alpha_promoted else "REJECTED"
                    print(f"  Alpha WR: {alpha_wr:.1%} -> {status}", file=sys.stderr)

                if alpha_promoted:
                    if alpha_candidate_pt.exists():
                        shutil.copy2(str(alpha_candidate_pt), str(alpha_best_pt))
                    shutil.copy2(str(alpha_candidate_onnx), str(alpha_best_onnx))
            else:
                print(f"  Alpha training failed (exit {rc})", file=sys.stderr)
                alpha_wr = 0.0
                alpha_promoted = False
        else:
            alpha_wr = 0.0
            alpha_promoted = False

        # ---- Phase 6: Train NNUE ----
        nnue_data_files = sorted(nnue_data_dir.glob("cross_*.npz"))
        if len(nnue_data_files) > 30:
            nnue_data_files = nnue_data_files[-30:]

        if nnue_data_files:
            print("\n--- Phase 6: Train NNUE ---", file=sys.stderr)
            nnue_candidate_pt = nnue_models_dir / "candidate.pt"
            nnue_candidate_bin = nnue_models_dir / "candidate.bin"

            train_args = [
                py, os.path.join(tools_dir, "train_nnue.py"),
                "--data", str(nnue_data_files[-1]),
                "--output", str(nnue_candidate_bin),
                "--model-pt", str(nnue_candidate_pt),
                "--epochs", str(args.nnue_train_epochs),
                "--batch-size", str(args.nnue_batch_size),
                "--lr", str(args.nnue_lr),
            ]
            if nnue_best_pt.exists():
                train_args += ["--resume", str(nnue_best_pt)]

            rc = run_cmd(train_args, f"NNUE training ({len(nnue_data_files)} data files)")

            if rc == 0 and nnue_candidate_bin.exists():
                # ---- Phase 8: Gate NNUE ----
                nnue_promoted = False
                if args.skip_eval or not nnue_best_bin.exists():
                    nnue_promoted = True
                    nnue_wr = 1.0
                    print("  NNUE promoted (no baseline or skip-eval)", file=sys.stderr)
                else:
                    print(f"\n--- Gate NNUE ({args.eval_games} games) ---", file=sys.stderr)
                    nnue_wr = evaluate_nnue(
                        args.nnue_engine, str(nnue_candidate_bin), str(nnue_best_bin),
                        args.eval_games, args.eval_movetime,
                    )
                    nnue_promoted = nnue_wr >= args.gate_threshold
                    status = "PROMOTED" if nnue_promoted else "REJECTED"
                    print(f"  NNUE WR: {nnue_wr:.1%} -> {status}", file=sys.stderr)

                if nnue_promoted:
                    shutil.copy2(str(nnue_candidate_pt), str(nnue_best_pt))
                    shutil.copy2(str(nnue_candidate_bin), str(nnue_best_bin))
            else:
                print(f"  NNUE training failed (exit {rc})", file=sys.stderr)
                nnue_wr = 0.0
                nnue_promoted = False
        else:
            nnue_wr = 0.0
            nnue_promoted = False

        # ---- Log ----
        iter_elapsed = time.time() - iter_start
        log_entry = {
            "iteration": iteration + 1,
            "alpha_positions": sum(len(p) for p in all_alpha_positions),
            "nnue_positions": len(all_nnue_samples),
            "alpha_wr": round(alpha_wr, 4) if alpha_data_files else None,
            "alpha_promoted": alpha_promoted,
            "nnue_wr": round(nnue_wr, 4) if nnue_data_files else None,
            "nnue_promoted": nnue_promoted,
            "elapsed_sec": round(iter_elapsed),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"\n  Iteration {iteration + 1} complete ({iter_elapsed:.0f}s). "
            f"Alpha: WR={alpha_wr:.1%} {'PROMOTED' if alpha_promoted else 'rejected'} | "
            f"NNUE: WR={nnue_wr:.1%} {'PROMOTED' if nnue_promoted else 'rejected'}",
            file=sys.stderr,
        )

        # Cleanup old Alpha candidates
        for old in alpha_models_dir.glob("candidate_*.pt"):
            if not alpha_data_files or old.stem != f"candidate_{iteration:04d}":
                old.unlink(missing_ok=True)
        for old in alpha_models_dir.glob("candidate_*.onnx"):
            if not alpha_data_files or old.stem != f"candidate_{iteration:04d}":
                old.unlink(missing_ok=True)

    print("\nCross-training loop complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
