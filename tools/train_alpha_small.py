#!/usr/bin/env python3
"""Train a smaller ResNet-SE model via knowledge distillation from the full model.

The student (10 blocks, 128ch, ~2-3M params) learns from the teacher's soft outputs,
producing a fast model suitable for CPU inference (~50ms/eval).

Usage:
  python train_alpha_small.py --teacher alpha_model.pt --data train_data.npz --output alpha_model_small
  python train_alpha_small.py --teacher alpha_model.pt --data sp*.npz --student alpha_model_small.pt --resume
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


STUDENT_CHANNELS = 128
STUDENT_BLOCKS = 10
TEMPERATURE = 3.0
DISTILL_ALPHA = 0.7  # weight for soft targets vs hard targets


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation for small Alpha model")
    parser.add_argument("--teacher", required=True, help="Teacher model (.pt)")
    parser.add_argument("--student", default=None, help="Student model to resume from (.pt)")
    parser.add_argument("--data", nargs="+", required=True, help=".npz training data files")
    parser.add_argument("--output", default="alpha_model_small", help="Output prefix (no extension)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=DISTILL_ALPHA,
                        help="Soft vs hard target weight (0=hard only, 1=soft only)")
    parser.add_argument("--teacher-channels", type=int, default=192)
    parser.add_argument("--teacher-blocks", type=int, default=15)
    parser.add_argument("--student-channels", type=int, default=STUDENT_CHANNELS)
    parser.add_argument("--student-blocks", type=int, default=STUDENT_BLOCKS)
    parser.add_argument("--buffer-size", type=int, default=2000000)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from train_alpha import build_model, pick_device, encoded_to_spatial, INPUT_CHANNELS

    device = pick_device(torch, args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Load data
    all_encoded = []
    all_policy = []
    all_wdl = []
    for data_path in args.data:
        print(f"Loading {data_path}...", end="", file=sys.stderr, flush=True)
        data = np.load(data_path)
        all_encoded.append(data["encoded"])
        if "policy_target" in data:
            all_policy.append(data["policy_target"])
        elif "policy" in data:
            all_policy.append(data["policy"])
        if "wdl_target" in data:
            all_wdl.append(data["wdl_target"])
        elif "wdl" in data:
            all_wdl.append(data["wdl"])
        elif "result" in data:
            r = data["result"]
            w = np.zeros((len(r), 3), dtype=np.float32)
            w[r == 1, 0] = 1.0
            w[r == 0, 1] = 1.0
            w[r == -1, 2] = 1.0
            all_wdl.append(w)
        print(f" {len(data['encoded'])} positions", file=sys.stderr)

    encoded = np.concatenate(all_encoded, axis=0)
    if all_policy:
        hard_policy = np.concatenate(all_policy, axis=0)
    else:
        hard_policy = None
    hard_wdl = np.concatenate(all_wdl, axis=0) if all_wdl else None
    del all_encoded, all_policy, all_wdl

    if len(encoded) > args.buffer_size:
        encoded = encoded[-args.buffer_size:]
        if hard_policy is not None:
            hard_policy = hard_policy[-args.buffer_size:]
        if hard_wdl is not None:
            hard_wdl = hard_wdl[-args.buffer_size:]

    n = len(encoded)
    print(f"Training data: {n:,} positions", file=sys.stderr)

    print("Converting to spatial features...", end="", file=sys.stderr, flush=True)
    t0 = time.time()
    features_np = encoded_to_spatial(encoded.astype(np.int16))
    del encoded
    print(f" {time.time() - t0:.1f}s", file=sys.stderr)

    features_t = torch.from_numpy(features_np)
    hard_policy_t = torch.from_numpy(hard_policy) if hard_policy is not None else None
    hard_wdl_t = torch.from_numpy(hard_wdl) if hard_wdl is not None else None
    del features_np, hard_policy, hard_wdl

    # Build teacher (frozen)
    teacher = build_model(nn, channels=args.teacher_channels,
                          num_blocks=args.teacher_blocks).to(device)
    teacher_state = torch.load(args.teacher, map_location=device, weights_only=True)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {t_params/1e6:.1f}M params ({args.teacher_channels}ch x {args.teacher_blocks}blk)",
          file=sys.stderr)

    # Build student
    student = build_model(nn, channels=args.student_channels,
                          num_blocks=args.student_blocks).to(device)
    if args.student and Path(args.student).exists():
        student.load_state_dict(torch.load(args.student, map_location=device, weights_only=True))
        print(f"Resumed student from {args.student}", file=sys.stderr)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {s_params/1e6:.1f}M params ({args.student_channels}ch x {args.student_blocks}blk)",
          file=sys.stderr)
    print(f"Compression: {t_params/s_params:.1f}x", file=sys.stderr)

    # Training
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    total_batches = (n + args.batch_size - 1) // args.batch_size
    total_steps = total_batches * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy='cos', div_factor=10, final_div_factor=100,
    )

    T = args.temperature
    alpha = args.alpha
    wdl_loss_fn = nn.CrossEntropyLoss()

    print(f"Distillation: T={T}, alpha={alpha} (soft={alpha:.0%}, hard={1-alpha:.0%})",
          file=sys.stderr)
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr:.2e}",
          file=sys.stderr)

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_loss_v = 0.0
        total_loss_p = 0.0
        batches = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            b_feat = features_t[idx].to(device, non_blocking=True)

            with torch.no_grad():
                teacher_wdl, teacher_policy = teacher(b_feat)

            student_wdl, student_policy = student(b_feat)

            # --- Policy distillation ---
            soft_teacher_p = F.softmax(teacher_policy / T, dim=1)
            log_student_p = F.log_softmax(student_policy / T, dim=1)
            loss_p_soft = -(soft_teacher_p * log_student_p).sum(dim=1).mean() * (T * T)

            if hard_policy_t is not None:
                b_hard_p = hard_policy_t[idx].to(device, non_blocking=True)
                log_student_p_hard = F.log_softmax(student_policy, dim=1)
                loss_p_hard = -(b_hard_p * log_student_p_hard).sum(dim=1).mean()
                loss_p = alpha * loss_p_soft + (1 - alpha) * loss_p_hard
            else:
                loss_p = loss_p_soft

            # --- Value distillation ---
            soft_teacher_v = F.softmax(teacher_wdl / T, dim=1)
            log_student_v = F.log_softmax(student_wdl / T, dim=1)
            loss_v_soft = -(soft_teacher_v * log_student_v).sum(dim=1).mean() * (T * T)

            if hard_wdl_t is not None:
                b_hard_wdl = hard_wdl_t[idx].to(device, non_blocking=True)
                wdl_idx = b_hard_wdl.argmax(dim=1)
                loss_v_hard = wdl_loss_fn(student_wdl, wdl_idx)
                loss_v = alpha * loss_v_soft + (1 - alpha) * loss_v_hard
            else:
                loss_v = loss_v_soft

            loss = loss_v + loss_p

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
            batches += 1

            pct = batches * 100 // total_batches
            filled = pct // 5
            bar = "=" * filled + ">" * (1 if filled < 20 else 0) + " " * (20 - filled - (1 if filled < 20 else 0))
            print(f"\r  epoch {epoch + 1}/{args.epochs}: [{bar}] {pct:3d}% | "
                  f"v={total_loss_v / batches:.4f} p={total_loss_p / batches:.4f}",
                  end="", file=sys.stderr, flush=True)

        elapsed = time.time() - t0
        avg_v = total_loss_v / batches
        avg_p = total_loss_p / batches
        avg_total = total_loss / batches
        lr = optimizer.param_groups[0]['lr']
        print(f"\r  epoch {epoch + 1}/{args.epochs}: "
              f"loss={avg_total:.4f} (v={avg_v:.4f} p={avg_p:.4f}) "
              f"lr={lr:.6f} ({elapsed:.1f}s)          ", file=sys.stderr)

    # Save
    output_pt = Path(f"{args.output}.pt")
    output_onnx = Path(f"{args.output}.onnx")

    if output_pt.parent != Path(""):
        output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), output_pt)
    print(f"Saved student model to {output_pt}", file=sys.stderr)

    # Export ONNX
    student = student.cpu()
    student.eval()
    dummy = torch.zeros(1, INPUT_CHANNELS, 9, 9)
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")
        torch.onnx.export(
            student, dummy, str(output_onnx),
            dynamo=False, opset_version=17,
            input_names=["features"],
            output_names=["value_wdl", "policy_logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "value_wdl": {0: "batch"},
                "policy_logits": {0: "batch"},
            },
        )
        print(f"ONNX exported: {output_onnx}", file=sys.stderr)

        orig_size = output_pt.stat().st_size / (1024 * 1024)
        onnx_size = output_onnx.stat().st_size / (1024 * 1024)
        print(f"  PyTorch: {orig_size:.1f} MB, ONNX: {onnx_size:.1f} MB", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}", file=sys.stderr)

    print(f"value_loss={avg_v:.4f} policy_loss={avg_p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
