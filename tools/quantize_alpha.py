#!/usr/bin/env python3
"""INT8 static quantization for Alpha engine ONNX model.

Produces a quantized model for fast CPU inference with minimal accuracy loss.

Usage:
  python quantize_alpha.py --model alpha_model.onnx --output alpha_model_int8.onnx
  python quantize_alpha.py --model alpha_model.onnx --output alpha_model_int8.onnx --calibration-data train_data.npz --verify
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize Alpha ONNX model")
    parser.add_argument("--model", required=True, help="Input ONNX model")
    parser.add_argument("--output", default=None, help="Output quantized model")
    parser.add_argument("--calibration-data", default=None,
                        help=".npz with encoded positions for calibration")
    parser.add_argument("--calibration-size", type=int, default=1000,
                        help="Number of positions for calibration")
    parser.add_argument("--verify", action="store_true",
                        help="Verify quantized model accuracy vs original")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.model)
        args.output = str(p.with_stem(p.stem + "_int8"))

    try:
        import onnxruntime
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
    except ImportError:
        print("ERROR: onnxruntime-quantization required. "
              "Install: pip install onnxruntime onnxruntime-quantization", file=sys.stderr)
        return 1

    # Generate or load calibration data
    if args.calibration_data:
        from train_alpha import encoded_to_spatial
        print(f"Loading calibration data from {args.calibration_data}...", file=sys.stderr)
        data = np.load(args.calibration_data)
        encoded = data["encoded"][:args.calibration_size].astype(np.int16)
        features = encoded_to_spatial(encoded)
    else:
        print(f"Generating random calibration data ({args.calibration_size} samples)...",
              file=sys.stderr)
        features = np.random.randn(args.calibration_size, 45, 9, 9).astype(np.float32)

    class ShogiCalibrationReader(CalibrationDataReader):
        def __init__(self, features: np.ndarray, batch_size: int = 16):
            self.features = features
            self.batch_size = batch_size
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.features):
                return None
            end = min(self.idx + self.batch_size, len(self.features))
            batch = self.features[self.idx:end]
            self.idx = end
            return {"features": batch}

        def rewind(self):
            self.idx = 0

    calibration_reader = ShogiCalibrationReader(features)

    print(f"Quantizing {args.model} -> {args.output}...", file=sys.stderr)
    t0 = time.time()

    try:
        quantize_static(
            args.model,
            args.output,
            calibration_reader,
            quant_format=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    except TypeError:
        quantize_static(
            args.model,
            args.output,
            calibration_reader,
        )

    elapsed = time.time() - t0
    orig_size = Path(args.model).stat().st_size / (1024 * 1024)
    quant_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Quantization complete ({elapsed:.1f}s)", file=sys.stderr)
    print(f"  Original:   {orig_size:.1f} MB", file=sys.stderr)
    print(f"  Quantized:  {quant_size:.1f} MB ({quant_size/orig_size*100:.0f}%)", file=sys.stderr)

    if args.verify:
        print("\nVerifying quantized model...", file=sys.stderr)
        verify_size = min(200, len(features))
        verify_features = features[:verify_size]

        sess_orig = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        sess_quant = onnxruntime.InferenceSession(args.output, providers=["CPUExecutionProvider"])

        policy_agree = 0
        value_mae = 0.0

        for i in range(0, verify_size, 16):
            batch = verify_features[i:min(i+16, verify_size)]
            feed = {"features": batch}

            orig_out = sess_orig.run(None, feed)
            quant_out = sess_quant.run(None, feed)

            orig_wdl = orig_out[0]
            quant_wdl = quant_out[0]
            orig_policy = orig_out[1]
            quant_policy = quant_out[1]

            orig_value = orig_wdl[:, 0] - orig_wdl[:, 2]
            quant_value = quant_wdl[:, 0] - quant_wdl[:, 2]
            value_mae += np.abs(orig_value - quant_value).sum()

            policy_agree += (orig_policy.argmax(axis=1) == quant_policy.argmax(axis=1)).sum()

        policy_accuracy = policy_agree / verify_size * 100
        value_mae /= verify_size
        print(f"  Policy top-1 agreement: {policy_accuracy:.1f}%", file=sys.stderr)
        print(f"  Value MAE:              {value_mae:.4f}", file=sys.stderr)

        if policy_accuracy < 90:
            print("  WARNING: Low policy agreement — quantization may be too aggressive",
                  file=sys.stderr)
        if value_mae > 0.05:
            print("  WARNING: High value MAE — quantization may hurt strength",
                  file=sys.stderr)

    print(f"size_mb={quant_size:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
