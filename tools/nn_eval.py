#!/usr/bin/env python3
"""Transformer-based shogi evaluator with persistent serve mode.

Board state protocol:
  81 square values (0=empty, 1-14=black pieces, 15-28=white pieces)
  + 7 black hand counts (Pawn..Rook)
  + 7 white hand counts (Pawn..Rook)
  + 1 side (0=black, 1=white)
  = 96 integers, space-separated.

Policy output: 2187 floats (81 squares * 27 channels).
  Channels 0-9: direction without promotion
  Channels 10-19: direction with promotion
  Channels 20-26: drop (Pawn..Rook)

Serve protocol (stdin/stdout, line-based):
  -> ready
  <- eval <96 ints>
  -> <value> <2187 policy floats>
  <- batch <N> | <96 ints> | <96 ints> | ...
  -> <value> <2187 policy floats>  (one line per board)
  <- train <path> <epochs>
  -> ok
  <- quit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

BOARD_SQUARES = 81
HAND_TYPES = 7
VOCAB_SIZE = 29       # 0=empty, 1-14=black, 15-28=white
HAND_MAX = 19         # max hand count per type (pawns)
SEQ_LEN = 95          # 81 board + 14 hand slots
POLICY_SIZE = 2187    # 81 * 27
STATE_SIZE = 96       # 81 + 7 + 7 + 1


def import_torch():
    import torch
    from torch import nn
    return torch, nn


def pick_device(torch_mod, requested: str):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


class ShogiTransformer:
    """Wrapper that builds the model lazily (after torch is imported)."""

    @staticmethod
    def build(nn, d_model=128, nhead=8, num_layers=4, dim_ff=256):
        return ShogiTransformerModel(nn, d_model, nhead, num_layers, dim_ff)


class ShogiTransformerModel:
    """Pure nn.Module-based transformer for shogi evaluation."""

    def __new__(cls, nn, d_model, nhead, num_layers, dim_ff):
        import torch
        import torch.nn.functional as F

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = d_model
                self.piece_embed = nn.Embedding(VOCAB_SIZE, d_model)
                self.file_embed = nn.Embedding(9, d_model)
                self.rank_embed = nn.Embedding(9, d_model)
                self.hand_pos_embed = nn.Embedding(2 * HAND_TYPES, d_model)
                self.hand_count_embed = nn.Embedding(HAND_MAX + 1, d_model)
                self.side_embed = nn.Embedding(2, d_model)

                self.cnn = nn.Sequential(
                    nn.Conv2d(d_model, d_model, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(d_model, d_model, 3, padding=1),
                )

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                    batch_first=True, dropout=0.1,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.value_head = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Tanh(),
                )

                self.policy_head = nn.Sequential(
                    nn.Linear(d_model * BOARD_SQUARES, 512),
                    nn.ReLU(),
                    nn.Linear(512, POLICY_SIZE),
                )

            def forward(self, board_tokens, hand_tokens, side_token):
                B = board_tokens.shape[0]
                dev = board_tokens.device
                d = self.d_model

                board_emb = self.piece_embed(board_tokens)
                files = torch.arange(BOARD_SQUARES, device=dev) % 9
                ranks = torch.arange(BOARD_SQUARES, device=dev) // 9
                board_emb = board_emb + self.file_embed(files) + self.rank_embed(ranks)

                residual = board_emb
                x = board_emb.transpose(1, 2).reshape(B, d, 9, 9)
                x = self.cnn(x)
                x = x.reshape(B, d, BOARD_SQUARES).transpose(1, 2)
                board_emb = F.relu(x + residual)

                hand_emb = self.hand_count_embed(hand_tokens)
                hand_ids = torch.arange(2 * HAND_TYPES, device=dev)
                hand_emb = hand_emb + self.hand_pos_embed(hand_ids)

                seq = torch.cat([board_emb, hand_emb], dim=1)
                seq = seq + self.side_embed(side_token).unsqueeze(1)

                out = self.transformer(seq)

                global_repr = out.mean(dim=1)
                value = self.value_head(global_repr).squeeze(-1)

                board_out = out[:, :BOARD_SQUARES, :].reshape(B, -1)
                policy_logits = self.policy_head(board_out)

                return value, policy_logits

        return _Model()


def parse_board_state(tokens: List[str]) -> Tuple[List[int], List[int], int]:
    """Parse 96 integers into (squares[81], hand[14], side)."""
    if len(tokens) < STATE_SIZE:
        raise ValueError(f"expected {STATE_SIZE} tokens, got {len(tokens)}")
    vals = [int(t) for t in tokens[:STATE_SIZE]]
    squares = vals[:BOARD_SQUARES]
    hand = vals[BOARD_SQUARES:BOARD_SQUARES + 2 * HAND_TYPES]
    side = vals[-1]
    return squares, hand, side


def boards_to_tensors(boards_data, torch_mod, device):
    """Convert list of (squares, hand, side) to tensors."""
    B = len(boards_data)
    board_t = torch_mod.zeros(B, BOARD_SQUARES, dtype=torch_mod.long, device=device)
    hand_t = torch_mod.zeros(B, 2 * HAND_TYPES, dtype=torch_mod.long, device=device)
    side_t = torch_mod.zeros(B, dtype=torch_mod.long, device=device)

    for i, (squares, hand, side) in enumerate(boards_data):
        for j, s in enumerate(squares):
            board_t[i, j] = max(0, min(s, VOCAB_SIZE - 1))
        for j, h in enumerate(hand):
            hand_t[i, j] = max(0, min(h, HAND_MAX))
        side_t[i] = side

    return board_t, hand_t, side_t


def softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def load_model(model_path, torch_mod, nn, device, require_exists=True):
    model = ShogiTransformer.build(nn).to(device)
    if not model_path.exists():
        if require_exists:
            raise FileNotFoundError(model_path)
        return model
    state = torch_mod.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


# ---------- Serve mode ----------

def _log(msg):
    print(f"[NN] {msg}", file=sys.stderr, flush=True)


def serve(args):
    torch_mod, nn = import_torch()
    device = pick_device(torch_mod, args.device)
    model_path = Path(args.model)
    _log(f"Device: {device}")

    model = load_model(model_path, torch_mod, nn, device, require_exists=False)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    if model_path.exists():
        _log(f"Model loaded: {model_path} ({params:,} params)")
    else:
        _log(f"No model file at {model_path}, using random weights ({params:,} params)")

    eval_count = 0

    sys.stdout.write("ready\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "quit":
            _log(f"Shutting down (evaluated {eval_count} positions total)")
            break

        if line.startswith("eval "):
            tokens = line[5:].split()
            try:
                squares, hand, side = parse_board_state(tokens)
                bt, ht, st = boards_to_tensors([(squares, hand, side)], torch_mod, device)
                with torch_mod.no_grad():
                    value, policy_logits = model(bt, ht, st)
                v = value.item()
                p = softmax(policy_logits[0].cpu().tolist())
                out_parts = [f"{v:.6f}"] + [f"{x:.6f}" for x in p]
                sys.stdout.write(" ".join(out_parts) + "\n")
                eval_count += 1
            except Exception as e:
                _log(f"Error in eval: {e}")
                sys.stdout.write(f"0.0 " + " ".join(["0.000460"] * POLICY_SIZE) + "\n")
            sys.stdout.flush()
            if eval_count % 100 == 0:
                _log(f"Evaluated {eval_count} positions")

        elif line.startswith("batch "):
            parts = line[6:].split("|")
            try:
                n = int(parts[0].strip())
                boards_data = []
                for i in range(1, n + 1):
                    tokens = parts[i].strip().split()
                    squares, hand, side = parse_board_state(tokens)
                    boards_data.append((squares, hand, side))

                bt, ht, st = boards_to_tensors(boards_data, torch_mod, device)
                with torch_mod.no_grad():
                    values, policy_logits = model(bt, ht, st)

                for i in range(n):
                    v = values[i].item()
                    p = softmax(policy_logits[i].cpu().tolist())
                    out_parts = [f"{v:.6f}"] + [f"{x:.6f}" for x in p]
                    sys.stdout.write(" ".join(out_parts) + "\n")
                sys.stdout.flush()
                eval_count += n
                if eval_count % 100 < n:
                    _log(f"Evaluated {eval_count} positions")
            except Exception as e:
                _log(f"Error in batch: {e}")
                for _ in range(n if 'n' in dir() else 1):
                    sys.stdout.write(f"0.0 " + " ".join(["0.000460"] * POLICY_SIZE) + "\n")
                sys.stdout.flush()

        elif line.startswith("train "):
            parts = line[6:].strip().split()
            data_path = parts[0]
            epochs = int(parts[1]) if len(parts) > 1 else 10
            _log(f"Training from {data_path} for {epochs} epochs")
            try:
                _train_from_file(model, data_path, epochs, torch_mod, nn, device, model_path)
                _log(f"Training complete, model saved to {model_path}")
                sys.stdout.write("ok\n")
            except Exception as e:
                _log(f"Training error: {e}")
                sys.stdout.write(f"error {e}\n")
            sys.stdout.flush()

    return 0


def _train_from_file(model, data_path, epochs, torch_mod, nn, device, model_path):
    """Train from self-play data file.

    Format per line: value_label <96 board state ints> <policy target: move_index>
    """
    import torch

    values = []
    boards_data = []
    policy_targets = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < STATE_SIZE + 2:
                continue
            v = float(parts[0])
            state_tokens = parts[1:STATE_SIZE + 1]
            move_idx = int(parts[STATE_SIZE + 1])
            squares, hand, side = parse_board_state(state_tokens)
            values.append(v)
            boards_data.append((squares, hand, side))
            policy_targets.append(move_idx)

    if not boards_data:
        return

    bt, ht, st = boards_to_tensors(boards_data, torch_mod, device)
    value_targets = torch_mod.tensor(values, dtype=torch_mod.float32, device=device)
    policy_idx = torch_mod.tensor(policy_targets, dtype=torch_mod.long, device=device)

    model.train()
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    n = bt.shape[0]
    batch_size = min(256, n)

    for epoch in range(epochs):
        perm = torch_mod.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred_v, pred_p = model(bt[idx], ht[idx], st[idx])
            loss_v = value_loss_fn(pred_v, value_targets[idx])
            loss_p = policy_loss_fn(pred_p, policy_idx[idx])
            loss = loss_v + loss_p
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True) if model_path.parent != Path("") else None
    torch_mod.save(model.state_dict(), model_path)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--model", default="nn_model.pt")
    serve_parser.add_argument("--device", default="auto")
    serve_parser.set_defaults(func=serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
