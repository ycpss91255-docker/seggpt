#!/usr/bin/env python3
"""Phase 0 benchmark CLI — exercise ``SegGPTBackend.infer()`` end-to-end.

Loads a target image + N reference (image, mask) pairs from disk,
calls Layer 2's ``SegGPTBackend.infer()``, and prints the raw output's
shape + telemetry. If an expected mask is provided, also reports
mIoU vs the prediction.

Default arguments point at the in-repo ``test/assets/hmbb`` fixtures
so a no-arg invocation runs a smoke test of the whole pipeline:

    python scripts/phase0.py

Phase 0 real prompts can be swapped in via the CLI:

    python scripts/phase0.py \\
        --target  prompts/<scene>/target.png \\
        --refs    prompts/<scene>/ref_1.jpg  prompts/<scene>/ref_2.jpg \\
        --masks   prompts/<scene>/mask_1.png prompts/<scene>/mask_2.png \\
        --expected prompts/<scene>/gt.png \\
        --save-mask out/mask.png

Output: a JSON record on stdout with shape / latency / gpu_mem / mIoU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "model" / "seggpt_vit_large.pth"
_DEFAULT_CONFIG = _REPO_ROOT / "model" / "seggpt_vit_large.yaml"
_DEFAULT_HMBB = _REPO_ROOT / "test" / "assets" / "hmbb"
_DEFAULT_EXPECTED = _REPO_ROOT / "test" / "assets" / "expected" / "output_hmbb_3.png"


def _read_rgb(path: Path) -> np.ndarray:
    """OpenCV reads BGR; flip to RGB for SegGPT's RGB contract."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    """Read a single-channel mask (grayscale)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def _compute_miou(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    """Mean-IoU between two binary masks. Both must be (H, W) or broadcastable."""
    pred_b = pred.astype(bool)
    true_b = true.astype(bool)
    intersection = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    return float(intersection) / float(union + eps)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL,
        help="Checkpoint path (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Model architecture YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=_DEFAULT_HMBB / "hmbb_3.jpg",
        help="Target RGB image (default: %(default)s).",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        nargs="+",
        default=[_DEFAULT_HMBB / "hmbb_1.jpg", _DEFAULT_HMBB / "hmbb_2.jpg"],
        help="Reference RGB image paths (default: hmbb_1, hmbb_2).",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        nargs="+",
        default=[_DEFAULT_HMBB / "hmbb_1_target.png", _DEFAULT_HMBB / "hmbb_2_target.png"],
        help="Reference mask paths (default: hmbb_1_target, hmbb_2_target).",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=_DEFAULT_EXPECTED,
        help="Optional ground-truth mask for mIoU. Pass empty string to skip.",
    )
    parser.add_argument(
        "--mode",
        choices=("instance", "semantic"),
        default="instance",
        help="Segmentation mode (default: %(default)s).",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="If given, write the predicted mask as a PNG.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warm-up forward to JIT CUDA kernels before timing.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.refs) != len(args.masks):
        raise SystemExit(
            f"--refs ({len(args.refs)}) and --masks ({len(args.masks)}) length mismatch"
        )
    for path in (args.model, args.config, args.target, *args.refs, *args.masks):
        if not path.exists():
            raise SystemExit(f"path not found: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)

    # Heavy imports deferred until after argparse: lets `--help` run on
    # any host without paying the torch / detectron2 import cost.
    from seggpt.api import SegGPTBackend

    print(f"[phase0] loading model: {args.model}", file=sys.stderr)
    t0 = time.perf_counter()
    backend = SegGPTBackend(
        model_path=args.model,
        config_path=args.config,
        warmup=args.warmup,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[phase0] model load: {load_ms:.1f} ms", file=sys.stderr)

    target = _read_rgb(args.target)
    refs = [_read_rgb(p) for p in args.refs]
    masks = [_read_mask(p) for p in args.masks]

    print(
        f"[phase0] inferring: target {args.target.name} ({target.shape}), "
        f"{len(refs)} refs, mode={args.mode}",
        file=sys.stderr,
    )
    result = backend.infer(target, refs, masks, mode=args.mode)

    pred_mask = result["mask"]
    record = {
        "target": str(args.target),
        "refs": [str(p) for p in args.refs],
        "masks": [str(p) for p in args.masks],
        "mode": args.mode,
        "model_load_ms": round(load_ms, 2),
        "inference_latency_ms": round(float(result["inference_latency_ms"]), 2),
        "gpu_mem_mb": round(float(result["gpu_mem_mb"]), 2),
        "mask_shape": list(pred_mask.shape),
        "mask_positive_pixels": int(pred_mask.astype(bool).sum()),
        "class_id": result["class_id"].tolist(),
    }

    if args.expected and str(args.expected):
        if not args.expected.exists():
            print(f"[phase0] expected mask missing, skipping mIoU: {args.expected}", file=sys.stderr)
        else:
            true_mask = _read_mask(args.expected)
            # Layer 1 returns (C, H, W); take the first class slice for mIoU
            # in instance/semantic single-class scenarios.
            pred_for_iou = pred_mask[0] if pred_mask.ndim == 3 else pred_mask
            if pred_for_iou.shape != true_mask.shape:
                pred_for_iou = cv2.resize(
                    pred_for_iou.astype(np.uint8),
                    (true_mask.shape[1], true_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            record["miou"] = round(_compute_miou(pred_for_iou, true_mask), 4)

    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        # Save the first class slice as 0/255 PNG for visual sanity.
        out = pred_mask[0] if pred_mask.ndim == 3 else pred_mask
        cv2.imwrite(str(args.save_mask), (out.astype(bool) * 255).astype(np.uint8))
        record["saved_mask"] = str(args.save_mask)

    json.dump(record, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
