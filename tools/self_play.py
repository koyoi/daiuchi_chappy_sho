#!/usr/bin/env python3
"""Run USI self-play matches between two models.

Usage:
  python self_play.py --engine build/Release/kishi-to.exe --model1 v002.pt --model2 v001.pt
  python self_play.py --engine build/Release/kishi-to.exe --games 20 --movetime 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm


MAX_MOVES = 256
MOVE_TIMEOUT = 30


class USIEngine:
    def __init__(self, engine_path: str, model: Optional[str] = None,
                 python: Optional[str] = None, device: Optional[str] = None,
                 simulations: Optional[int] = None):
        self.engine_path = engine_path
        self.model = model
        self.python = python
        self.device = device
        self.simulations = simulations
        self.proc: Optional[subprocess.Popen] = None
        self.name = "unknown"

    def start(self):
        engine = str(Path(self.engine_path).resolve())
        self.proc = subprocess.Popen(
            [engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("usi")
        while True:
            line = self._recv()
            if line is None:
                raise RuntimeError(f"Engine {self.engine_path} died during USI init")
            if line.startswith("id name "):
                self.name = line[8:].strip()
            if line.strip() == "usiok":
                break

        self._send("setoption name RecordOnly value true")
        if self.model:
            self._send(f"setoption name NNModel value {self.model}")
        if self.python:
            self._send(f"setoption name NNPython value {self.python}")
        if self.device:
            self._send(f"setoption name NNDevice value {self.device}")
        if self.simulations is not None:
            self._send(f"setoption name MctsSimulations value {self.simulations}")

        self._send("isready")
        while True:
            line = self._recv()
            if line is None:
                raise RuntimeError(f"Engine {self.engine_path} died during isready")
            if line.strip() == "readyok":
                break

    def new_game(self):
        self._send("usinewgame")

    def go(self, position_cmd: str, movetime: int) -> str:
        self._send(position_cmd)
        self._send(f"go movetime {movetime}")
        deadline = time.monotonic() + MOVE_TIMEOUT + movetime / 1000.0
        while True:
            if time.monotonic() > deadline:
                return "resign"
            line = self._recv()
            if line is None:
                return "resign"
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "resign"

    def gameover(self, result: str):
        self._send(f"gameover {result}")

    def quit(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def _recv(self) -> Optional[str]:
        if not self.proc or not self.proc.stdout:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        return line.rstrip("\n\r")


def detect_repetition(moves: list[str]) -> bool:
    """4回同じ手順の繰り返しで千日手（標準ルールに準拠）"""
    if len(moves) < 16:
        return False
    for cycle_len in (4, 6, 8):
        need = cycle_len * 4
        if len(moves) < need:
            continue
        tail = moves[-need:]
        chunks = [tail[i * cycle_len:(i + 1) * cycle_len] for i in range(4)]
        if chunks[0] == chunks[1] == chunks[2] == chunks[3]:
            return True
    return False


def play_one_game(engine_black: USIEngine, engine_white: USIEngine,
                  movetime: int, game_label: str) -> tuple[str, int]:
    """Returns (result, ply_count)."""
    engine_black.new_game()
    engine_white.new_game()

    moves: list[str] = []

    pbar = tqdm(total=MAX_MOVES, desc=game_label, unit="mv",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                file=sys.stderr, leave=False, dynamic_ncols=True,
                miniters=1, mininterval=0.3)

    try:
        for ply in range(MAX_MOVES):
            current = engine_black if ply % 2 == 0 else engine_white

            if moves:
                pos_cmd = "position startpos moves " + " ".join(moves)
            else:
                pos_cmd = "position startpos"

            bestmove = current.go(pos_cmd, movetime)
            pbar.update(1)

            if bestmove == "resign" or bestmove == "none":
                if ply % 2 == 0:
                    engine_black.gameover("lose")
                    engine_white.gameover("win")
                    return "white", ply
                else:
                    engine_black.gameover("win")
                    engine_white.gameover("lose")
                    return "black", ply

            moves.append(bestmove)

            if detect_repetition(moves):
                engine_black.gameover("draw")
                engine_white.gameover("draw")
                return "draw(rep)", ply + 1
    finally:
        pbar.close()

    engine_black.gameover("draw")
    engine_white.gameover("draw")
    return "draw(max)", len(moves)


def run_match(engine_path: str, model1: Optional[str], model2: Optional[str],
              games: int, movetime: int, python: Optional[str],
              device: Optional[str], simulations: Optional[int] = None) -> dict:
    e1 = USIEngine(engine_path, model1, python, device, simulations)
    e2 = USIEngine(engine_path, model2, python, device, simulations)

    label1 = Path(model1).stem if model1 else "default"
    label2 = Path(model2).stem if model2 else "default"

    sim_info = f", {simulations}sims" if simulations else ""
    tqdm.write(f"Match: {label1} vs {label2} "
               f"({games} games, {movetime}ms/move{sim_info})",
               file=sys.stderr)

    e1.start()
    e2.start()

    wins1 = 0
    wins2 = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    match_start = time.monotonic()

    try:
        for game_idx in range(games):
            if game_idx % 2 == 0:
                black_engine, white_engine = e1, e2
                black_label, white_label = label1, label2
            else:
                black_engine, white_engine = e2, e1
                black_label, white_label = label2, label1

            game_label = (f"Game {game_idx+1}/{games} "
                          f"{black_label}(B) vs {white_label}(W)")

            game_start = time.monotonic()
            result, ply_count = play_one_game(
                black_engine, white_engine, movetime, game_label)
            game_elapsed = time.monotonic() - game_start

            if result == "black":
                winner_label = f"{black_label}(B)"
                black_wins += 1
                if black_label == label1:
                    wins1 += 1
                else:
                    wins2 += 1
            elif result == "white":
                winner_label = f"{white_label}(W)"
                white_wins += 1
                if white_label == label1:
                    wins1 += 1
                else:
                    wins2 += 1
            else:
                winner_label = result
                draws += 1

            decisive = black_wins + white_wins
            b_pct = f"{100.0*black_wins/decisive:.0f}" if decisive else "-"
            tqdm.write(
                f"  Game {game_idx+1}/{games}: "
                f"{black_label}(B) vs {white_label}(W) -> {winner_label} "
                f"({ply_count}moves, {game_elapsed:.0f}s) "
                f"[B:{black_wins} W:{white_wins} D:{draws} "
                f"B%={b_pct}%]",
                file=sys.stderr)
    finally:
        e1.quit()
        e2.quit()

    total = wins1 + wins2 + draws
    decisive = black_wins + white_wins
    win_rate = wins1 / total * 100 if total > 0 else 0
    b_pct = 100.0 * black_wins / decisive if decisive else 0
    total_elapsed = time.monotonic() - match_start
    color_stats = f"Black:{black_wins} White:{white_wins} Draw:{draws} (B%={b_pct:.1f}%)"
    if label1 != label2:
        model_stats = f"{label1} vs {label2}: {wins1}W-{wins2}L-{draws}D ({win_rate:.1f}%)"
    else:
        model_stats = f"{label1} self-play"
    summary = f"{model_stats} | {color_stats} | {total_elapsed:.0f}s"
    tqdm.write(summary, file=sys.stderr)

    return {
        "model1": label1,
        "model2": label2,
        "wins1": wins1,
        "wins2": wins2,
        "draws": draws,
        "win_rate": win_rate,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="USI self-play between two models")
    parser.add_argument("--engine", required=True, help="Path to USI engine executable")
    parser.add_argument("--model1", default=None, help="Model path for engine 1")
    parser.add_argument("--model2", default=None, help="Model path for engine 2")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--movetime", type=int, default=1000, help="Time per move (ms)")
    parser.add_argument("--python", default=None, help="Python path for NNPython option")
    parser.add_argument("--device", default=None, help="Device for NNDevice option")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (overrides engine default)")
    args = parser.parse_args()

    result = run_match(
        args.engine, args.model1, args.model2,
        args.games, args.movetime, args.python, args.device,
        args.simulations,
    )

    print(result["summary"])
    return 0 if result["win_rate"] >= 50.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
