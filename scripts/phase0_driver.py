#!/usr/bin/env python3
"""Phase 0 sub-flow driver — sweep targets × N, write run dir.

For each (target, N) pair:
  - pick N prompts from pool by diversity-ordered subset
  - run ``SegGPTBackend.infer()`` (model loaded once)
  - compute mIoU vs ``<gt-dir>/<target-filename>`` (if available)
  - save pred mask, copy failures (mIoU < threshold)

Outputs ``<run-dir-base>/<run-name>/``:
  meta.json, per_image.csv, stats.json, n_sweep.csv,
  pred_masks/<stem>_N<n>.png, failures/<stem>_N<n>/{target,pred,gt}.png,
  SUMMARY.md

Defaults: ``phase_0_test/{iron_beam_prompt,small_target,ground_truth}/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
# data/ is .gitignore'd evaluation data — mounted by virtue of the whole
# repo being mounted at ~/work, kept out of the backend repo to honour
# "SegGPT Backend Repo (職責邊界)" generic-reuse rule.
_DEFAULT_DATA = _REPO / "data" / "phase_0_test"
_DEFAULT_PROMPTS = _DEFAULT_DATA / "iron_beam_prompt"
_DEFAULT_TARGETS = _DEFAULT_DATA / "small_target"
_DEFAULT_GTS = _DEFAULT_DATA / "ground_truth"
_DEFAULT_MODEL = _REPO / "model" / "seggpt_vit_large.pth"
_DEFAULT_CONFIG = _REPO / "model" / "seggpt_vit_large.yaml"

# Diversity-ordered subsets indexed by prompt number (1..8).
# Source: phase_0_test/iron_beam_prompt/README.md.
#   01 純鐵 baseline   02 高解析+label   03 橘+綠
#   04 純綠            05 純橘            06 空棧板
#   07 強反光 hard     08 三色齊全
_N_SUBSETS: Dict[int, List[int]] = {
    1: [1],                        # pure iron baseline
    4: [1, 4, 5, 7],               # iron + green + orange + reflection
    8: [1, 2, 3, 4, 5, 6, 7, 8],   # full pool
}

_FailureThreshold = 0.5


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask: {path}")
    return mask


def _miou(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = pred.astype(bool)
    true_b = true.astype(bool)
    inter = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    return float(inter) / float(union + eps)


def _resize_to(pred: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    if pred.shape == shape_hw:
        return pred
    return cv2.resize(
        pred.astype(np.uint8),
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _index_prompts(prompts_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    pairs: Dict[int, Tuple[Path, Path]] = {}
    for img in sorted(prompts_dir.glob("prompt_*.png")):
        if "_mask" in img.stem:
            continue
        idx = int(img.stem.split("_")[1])
        mask = prompts_dir / f"{img.stem}_mask.png"
        if not mask.exists():
            raise SystemExit(f"missing mask for {img}: expected {mask}")
        pairs[idx] = (img, mask)
    return pairs


def _git_commit() -> str:
    res = subprocess.run(
        ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() or "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts-dir", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--targets-dir", type=Path, default=_DEFAULT_TARGETS)
    parser.add_argument("--gt-dir", type=Path, default=_DEFAULT_GTS)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output dir basename. Default: <YYYY-MM-DD_HHMM>_iron_beam_small_target.",
    )
    parser.add_argument(
        "--run-dir-base",
        type=Path,
        default=_REPO / "phase0_runs",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--mode", choices=("instance", "semantic"), default="instance")
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=_FailureThreshold,
        help="mIoU below this copies target/pred/gt to failures/.",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip mIoU (visual inspection / latency only).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    run_name = args.run_name or (
        datetime.now().strftime("%Y-%m-%d_%H%M") + "_iron_beam_small_target"
    )
    run_dir = args.run_dir_base / run_name
    (run_dir / "pred_masks").mkdir(parents=True, exist_ok=True)
    if not args.no_gt:
        (run_dir / "failures").mkdir(parents=True, exist_ok=True)

    prompt_pairs = _index_prompts(args.prompts_dir)
    for n in args.n_values:
        if n not in _N_SUBSETS:
            raise SystemExit(
                f"--n-values {n}: only {sorted(_N_SUBSETS)} are predefined; "
                f"edit _N_SUBSETS in {Path(__file__).name} to add more."
            )
        for idx in _N_SUBSETS[n]:
            if idx not in prompt_pairs:
                raise SystemExit(
                    f"N={n} requires prompt_{idx:02d} but {args.prompts_dir} "
                    f"only has {sorted(prompt_pairs)}"
                )

    targets = sorted(args.targets_dir.glob("*.png"))
    if not targets:
        raise SystemExit(f"no *.png in {args.targets_dir}")

    if not args.no_gt:
        missing = [t.name for t in targets if not (args.gt_dir / t.name).exists()]
        if missing:
            raise SystemExit(
                f"missing GT for {len(missing)} targets in {args.gt_dir}, "
                f"e.g. {missing[:3]}. Pass --no-gt to skip mIoU."
            )

    commit = _git_commit()
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "model_path": str(args.model),
        "config_path": str(args.config),
        "prompts_dir": str(args.prompts_dir),
        "targets_dir": str(args.targets_dir),
        "gt_dir": None if args.no_gt else str(args.gt_dir),
        "n_values": args.n_values,
        "n_subsets": {n: _N_SUBSETS[n] for n in args.n_values},
        "n_targets": len(targets),
        "mode": args.mode,
        "failure_threshold": args.failure_threshold,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[driver] loading model {args.model}", file=sys.stderr)
    t_load = time.perf_counter()
    from seggpt.api import SegGPTBackend  # heavy import deferred

    backend = SegGPTBackend(model_path=args.model, config_path=args.config)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    print(f"[driver] model loaded in {load_ms:.1f} ms", file=sys.stderr)

    rows: List[Dict] = []
    for tgt_path in targets:
        target_img = _read_rgb(tgt_path)
        gt_mask = None
        if not args.no_gt:
            gt_mask = _read_mask(args.gt_dir / tgt_path.name)

        for n in args.n_values:
            indices = _N_SUBSETS[n]
            refs = [_read_rgb(prompt_pairs[i][0]) for i in indices]
            masks = [_read_mask(prompt_pairs[i][1]) for i in indices]

            print(f"[driver] {tgt_path.name} N={n}", file=sys.stderr)
            result = backend.infer(target_img, refs, masks, mode=args.mode)
            pred = result["mask"]
            pred_first = pred[0] if pred.ndim == 3 else pred

            row = {
                "target": tgt_path.name,
                "N": n,
                "latency_ms": round(float(result["inference_latency_ms"]), 2),
                "gpu_mem_mb": round(float(result["gpu_mem_mb"]), 2),
                "mask_positive_pixels": int(pred_first.astype(bool).sum()),
            }

            out_pred = run_dir / "pred_masks" / f"{tgt_path.stem}_N{n}.png"
            cv2.imwrite(str(out_pred), (pred_first.astype(bool) * 255).astype(np.uint8))

            if gt_mask is not None:
                pred_for_iou = _resize_to(pred_first, gt_mask.shape)
                row["miou"] = round(_miou(pred_for_iou, gt_mask), 4)

                if row["miou"] < args.failure_threshold:
                    fail_dir = run_dir / "failures" / f"{tgt_path.stem}_N{n}"
                    fail_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(fail_dir / "target.png"), cv2.imread(str(tgt_path)))
                    cv2.imwrite(
                        str(fail_dir / "pred.png"),
                        (pred_for_iou.astype(bool) * 255).astype(np.uint8),
                    )
                    cv2.imwrite(str(fail_dir / "gt.png"), gt_mask)

            rows.append(row)

    fieldnames = ["target", "N", "latency_ms", "gpu_mem_mb", "mask_positive_pixels"]
    if not args.no_gt:
        fieldnames.append("miou")
    with (run_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats: Dict[str, Dict] = {}
    for n in args.n_values:
        rows_n = [r for r in rows if r["N"] == n]
        lat = np.array([r["latency_ms"] for r in rows_n])
        s: Dict = {
            "n_samples": len(rows_n),
            "latency_ms": {
                "median": float(np.median(lat)),
                "p10": float(np.percentile(lat, 10)),
                "p90": float(np.percentile(lat, 90)),
                "max": float(lat.max()),
            },
            "gpu_mem_mb_peak": float(
                max(r["gpu_mem_mb"] for r in rows_n)
            ),
        }
        if not args.no_gt:
            iou = np.array([r["miou"] for r in rows_n])
            s["miou"] = {
                "mean": float(np.mean(iou)),
                "median": float(np.median(iou)),
                "p10": float(np.percentile(iou, 10)),
                "p90": float(np.percentile(iou, 90)),
                "std": float(np.std(iou)),
            }
        stats[f"N={n}"] = s
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    with (run_dir / "n_sweep.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        cols = ["N", "median_latency_ms", "peak_gpu_mem_mb"]
        if not args.no_gt:
            cols.extend(["mean_miou", "median_miou", "p10_miou", "p90_miou"])
        writer.writerow(cols)
        for n in args.n_values:
            s = stats[f"N={n}"]
            row_out = [n, s["latency_ms"]["median"], s["gpu_mem_mb_peak"]]
            if not args.no_gt:
                iou = s["miou"]
                row_out.extend(
                    [iou["mean"], iou["median"], iou["p10"], iou["p90"]]
                )
            writer.writerow(row_out)

    _write_summary(run_dir, run_name, commit, args, targets, stats)
    print(f"[driver] done -> {run_dir}", file=sys.stderr)
    return 0


def _write_summary(
    run_dir: Path,
    run_name: str,
    commit: str,
    args: argparse.Namespace,
    targets: List[Path],
    stats: Dict[str, Dict],
) -> None:
    """Emit a markdown summary keyed off the main test-flow §7 thresholds."""
    lines = [
        f"# Phase 0 sub-flow run: {run_name}",
        "",
        f"- commit: `{commit}`",
        f"- targets: {len(targets)} (`{args.targets_dir}`)",
        f"- prompts: `{args.prompts_dir}`",
        f"- N sweep: {args.n_values}",
        f"- mode: {args.mode}",
        f"- gt: {'disabled' if args.no_gt else f'`{args.gt_dir}`'}",
        "",
        "## Per-N stats",
        "",
    ]
    for n in args.n_values:
        s = stats[f"N={n}"]
        lat = s["latency_ms"]
        lines.append(f"### N={n}")
        lines.append(
            f"- latency median: {lat['median']:.1f} ms "
            f"(p10 {lat['p10']:.1f} / p90 {lat['p90']:.1f} / max {lat['max']:.1f})"
        )
        lines.append(f"- peak GPU mem: {s['gpu_mem_mb_peak']:.0f} MB")
        if not args.no_gt:
            iou = s["miou"]
            lines.append(
                f"- mIoU: mean {iou['mean']:.3f} / median {iou['median']:.3f} "
                f"/ p10 {iou['p10']:.3f} / p90 {iou['p90']:.3f} / std {iou['std']:.3f}"
            )
        lines.append("")

    if not args.no_gt and 8 in args.n_values:
        s8 = stats["N=8"]
        gate = [
            ("mean(mIoU) >= 0.7 (N=8)", s8["miou"]["mean"], 0.7, ">="),
            ("p10(mIoU) >= 0.5 (N=8)", s8["miou"]["p10"], 0.5, ">="),
            ("latency median <= 500 ms (N=8)", s8["latency_ms"]["median"], 500, "<="),
            ("peak GPU mem <= 12000 MB (N=8)", s8["gpu_mem_mb_peak"], 12000, "<="),
        ]
        lines.append("## 通過判讀（主流程 §7）")
        lines.append("")
        lines.append("| 條件 | 值 | 通過 |")
        lines.append("|---|---|---|")
        for label, value, thresh, op in gate:
            ok = (value >= thresh) if op == ">=" else (value <= thresh)
            lines.append(f"| {label} | {value:.3f} | {'是' if ok else '否'} |")
        lines.append("")

    lines.append("## 下一步")
    lines.append("")
    lines.append("- 看 `pred_masks/` 並排確認邊界 ≤ 3 px（不串色 / 不框影子）。")
    if not args.no_gt:
        lines.append("- `failures/` 列 mIoU < 0.5 的 case，逐張看是否 prompt 多樣性不夠。")
    lines.append("- 全過 → 補綠 / 藍擋板 + 棧板下緣 prompt pool，擴大正式 Phase 0。")
    lines.append("- 任一條沒過 → 列 root cause（多樣性 / 場景 / 資源），決定調 pool / 換 backend / 改解析度。")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
