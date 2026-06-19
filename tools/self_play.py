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
from pathlib import Path
from typing import Optional


MAX_MOVES = 512


class USIEngine:
    def __init__(self, engine_path: str, model: Optional[str] = None,
                 python: Optional[str] = None, device: Optional[str] = None):
        self.engine_path = engine_path
        self.model = model
        self.python = python
        self.device = device
        self.proc: Optional[subprocess.Popen] = None
        self.name = "unknown"

    def start(self):
        self.proc = subprocess.Popen(
            [self.engine_path],
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

        if self.model:
            self._send(f"setoption name NNModel value {self.model}")
        if self.python:
            self._send(f"setoption name NNPython value {self.python}")
        if self.device:
            self._send(f"setoption name NNDevice value {self.device}")

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
        while True:
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
        if self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                return None
            return line.rstrip("\n\r")
        return None


def play_one_game(engine_black: USIEngine, engine_white: USIEngine,
                  movetime: int) -> str:
    engine_black.new_game()
    engine_white.new_game()

    moves: list[str] = []

    for ply in range(MAX_MOVES):
        current = engine_black if ply % 2 == 0 else engine_white

        if moves:
            pos_cmd = "position startpos moves " + " ".join(moves)
        else:
            pos_cmd = "position startpos"

        bestmove = current.go(pos_cmd, movetime)

        if bestmove == "resign":
            if ply % 2 == 0:
                engine_black.gameover("lose")
                engine_white.gameover("win")
                return "white"
            else:
                engine_black.gameover("win")
                engine_white.gameover("lose")
                return "black"

        moves.append(bestmove)

    engine_black.gameover("draw")
    engine_white.gameover("draw")
    return "draw"


def run_match(engine_path: str, model1: Optional[str], model2: Optional[str],
              games: int, movetime: int, python: Optional[str],
              device: Optional[str]) -> dict:
    e1 = USIEngine(engine_path, model1, python, device)
    e2 = USIEngine(engine_path, model2, python, device)

    label1 = Path(model1).stem if model1 else "default"
    label2 = Path(model2).stem if model2 else "default"

    print(f"Match: {label1} vs {label2} ({games} games, {movetime}ms/move)",
          file=sys.stderr)

    e1.start()
    e2.start()

    wins1 = 0
    wins2 = 0
    draws = 0

    try:
        for game_idx in range(games):
            if game_idx % 2 == 0:
                black_engine, white_engine = e1, e2
                black_label, white_label = label1, label2
            else:
                black_engine, white_engine = e2, e1
                black_label, white_label = label2, label1

            result = play_one_game(black_engine, white_engine, movetime)

            if result == "black":
                winner = black_label
                if black_label == label1:
                    wins1 += 1
                else:
                    wins2 += 1
            elif result == "white":
                winner = white_label
                if white_label == label1:
                    wins1 += 1
                else:
                    wins2 += 1
            else:
                winner = "draw"
                draws += 1

            print(f"  Game {game_idx + 1}/{games}: "
                  f"{black_label}(B) vs {white_label}(W) -> {winner}  "
                  f"[{label1} {wins1}W-{wins2}L-{draws}D]",
                  file=sys.stderr)
    finally:
        e1.quit()
        e2.quit()

    total = wins1 + wins2 + draws
    win_rate = wins1 / total * 100 if total > 0 else 0
    summary = f"{label1} vs {label2}: {wins1}W-{wins2}L-{draws}D ({win_rate:.1f}%)"
    print(summary, file=sys.stderr)

    return {
        "model1": label1,
        "model2": label2,
        "wins1": wins1,
        "wins2": wins2,
        "draws": draws,
        "win_rate": win_rate,
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
    args = parser.parse_args()

    result = run_match(
        args.engine, args.model1, args.model2,
        args.games, args.movetime, args.python, args.device,
    )

    print(result["summary"])
    return 0 if result["win_rate"] >= 50.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
