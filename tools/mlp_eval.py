#!/usr/bin/env python3
"""Small PyTorch scorer/trainer for the C++ shogi engine.

Input feature files are tab-separated rows with FeatureCount numeric columns.
Training data is tab-separated rows: label<TAB>feature_0<TAB>...<TAB>feature_n.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


FEATURE_COUNT = 100


def import_torch():
    import torch
    from torch import nn

    return torch, nn


def pick_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_model(nn):
    return nn.Sequential(
        nn.BatchNorm1d(FEATURE_COUNT),
        nn.Linear(FEATURE_COUNT, 256),
        nn.LeakyReLU(0.01),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.01),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )


def fuse_batchnorm_into_linear(bn_weight, bn_bias, bn_mean, bn_var,
                                bn_eps, linear_weight, linear_bias):
    """Fuse BatchNorm parameters into the following Linear layer.

    Returns (fused_weight, fused_bias) so the C++ side needs no BN logic.
    BN transform: y = (x - mean) / sqrt(var + eps) * gamma + beta
    Followed by Linear: z = W @ y + b
    Combined: z = W @ (gamma / sqrt(var+eps)) @ x + W @ (beta - gamma*mean/sqrt(var+eps)) + b
    Equivalent to a single Linear with modified weights/bias.
    """
    import torch
    scale = bn_weight / torch.sqrt(bn_var + bn_eps)
    shift = bn_bias - bn_mean * scale
    fused_w = linear_weight * scale.unsqueeze(0)
    fused_b = linear_bias + linear_weight @ shift
    return fused_w, fused_b


def load_features(path: Path, torch, device) -> "torch.Tensor":
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = [float(item) for item in line.split("\t")]
            if len(values) != FEATURE_COUNT:
                raise ValueError(f"expected {FEATURE_COUNT} features, got {len(values)}")
            rows.append(values)
    if not rows:
        return torch.empty((0, FEATURE_COUNT), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def load_training(path: Path, torch, device) -> Tuple["torch.Tensor", "torch.Tensor"]:
    labels = []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = [float(item) for item in line.split("\t")]
            if len(values) != FEATURE_COUNT + 1:
                continue
            labels.append(values[0])
            rows.append(values[1:])
    if not rows:
        raise ValueError("training data is empty")
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    return x, y


def load_model(model_path: Path, torch, nn, device, require_exists: bool):
    model = make_model(nn).to(device)
    if not model_path.exists():
        if require_exists:
            raise FileNotFoundError(model_path)
        return model
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        if require_exists:
            raise
        import sys
        print(f"Warning: checkpoint shape mismatch, starting fresh", file=sys.stderr)
        return model
    return model


def score(args) -> int:
    torch, nn = import_torch()
    device = pick_device(torch, args.device)
    x = load_features(Path(args.input), torch, device)
    model = load_model(Path(args.model), torch, nn, device, require_exists=True)
    model.eval()
    with torch.no_grad():
        if x.numel() == 0:
            scores = []
        else:
            logits = model(x).view(-1)
            wp = torch.sigmoid(logits)
            scores = ((wp - 0.5) * 2000.0).detach().cpu().tolist()
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for value in scores:
            handle.write(f"{value:.6f}\n")
    return 0


def train(args) -> int:
    torch, nn = import_torch()
    device = pick_device(torch, args.device)
    x, y = load_training(Path(args.data), torch, device)
    model_path = Path(args.model)
    model = load_model(model_path, torch, nn, device, require_exists=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    target = y.clamp(0.0, 1.0)

    import sys, time
    print(f"Training: {x.shape[0]} samples, {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}, device={device}", file=sys.stderr)

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(x.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        t0 = time.time()
        num_batches = (x.shape[0] + args.batch_size - 1) // args.batch_size
        for batch_i, start in enumerate(range(0, x.shape[0], args.batch_size)):
            index = permutation[start : start + args.batch_size]
            batch_x = x[index]
            batch_y = target[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            preds = (logits > 0).float()
            targets_binary = (batch_y > 0.5).float()
            correct += (preds == targets_binary).sum().item()
            total += batch_y.shape[0]
            if (batch_i + 1) % 100 == 0 or batch_i + 1 == num_batches:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / n_batches
                acc = 100.0 * correct / total if total > 0 else 0.0
                print(f"\r  epoch {epoch+1}/{args.epochs}: "
                      f"{batch_i+1}/{num_batches} batches | "
                      f"loss={avg_loss:.4f} acc={acc:.1f}% "
                      f"({elapsed:.1f}s)", end="", file=sys.stderr)
        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0
        lr_now = scheduler.get_last_lr()[0]
        print(f"\r  epoch {epoch+1}/{args.epochs}: "
              f"loss={avg_loss:.4f} acc={acc:.1f}% lr={lr_now:.2e} ({elapsed:.1f}s)          ",
              file=sys.stderr)

    model_path.parent.mkdir(parents=True, exist_ok=True) if model_path.parent != Path("") else None
    torch.save(model.state_dict(), model_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--model", default="mlp_model.pt")
    score_parser.add_argument("--device", default="auto")
    score_parser.set_defaults(func=score)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--model", default="mlp_model.pt")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
