#!/usr/bin/env python3
"""Phase 0 sub-flow driver — sweep targets × N, write run dir.

For each (target, N) pair:
  - pick the first N prompts (``prompt_01 .. prompt_0N``) from prompts-dir
  - run ``SegGPTBackend.infer()`` (model loaded once)
  - emit target original + overlay PNGs for visual inspection

Outputs ``<run-dir-base>/<run-name>/``:
  meta.json, per_image_N<n>.csv, stats.json, n_sweep.csv, SUMMARY.md,
  image/<target_stem>.png       (raw target, once per target),
  image/<target_stem>_N<n>.png  (target + tinted mask, one per N)

Workflow: prompts-dir is meant to be a symlink (``ln -sfn``) re-pointed
at whichever prompt set you want to evaluate. Each run writes a fresh
timestamped sub-dir under ``output/`` so swap-and-rerun keeps prior
results intact for side-by-side comparison.

GT mIoU / failures / pass-rate are gated by ``no_gt`` in the driver
YAML — default ``true`` until annotated GT lands. With no_gt the driver
emits only latency / GPU-mem stats.

Defaults: ``data/phase_0_test/{prompt,target,ground_truth}/``
(``prompt`` and ``target`` are typically symlinks managed by ``ln -sfn``).
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
# the SegGPT backend repo job-boundary (generic-reuse) rule.
_DEFAULT_DATA = _REPO / "data" / "phase_0_test"
# `prompt` is meant to be a symlink (`ln -sfn <prompt_set> prompt`)
# re-pointed before each run; the driver reads `prompt_01..prompt_08`
# from whatever the symlink resolves to right now.
_DEFAULT_PROMPTS = _DEFAULT_DATA / "prompt"
_DEFAULT_TARGETS = _DEFAULT_DATA / "target"
_DEFAULT_GTS = _DEFAULT_DATA / "ground_truth"
_DEFAULT_MODEL = _REPO / "model" / "seggpt_vit_large.pth"
_DEFAULT_CONFIG = _REPO / "model" / "seggpt_vit_large.yaml"
# Driver-level YAML (run params / overlay style). Separate from the
# SegGPT model architecture YAML above. CLI flags override every key
# present here; missing keys fall back to the hardcoded defaults below.
_DEFAULT_DRIVER_CONFIG = _REPO / "config" / "phase0_driver.yaml"

# Whitelist of YAML keys that may set parser defaults. Keep this in
# sync with _build_parser. Entries not in the whitelist raise — saves
# a surprising silent typo when someone edits the YAML.
_DRIVER_CONFIG_KEYS = {
    "n_values",
    "mode",
    "overlay_color",
    "overlay_alpha",
    "no_gt",
}

# N=k uses prompt_01..prompt_0k. The current workflow is to rotate
# `prompts_dir` itself via `ln -sfn`, so any "diversity-ordered subset"
# selection logic would conflict with the user's hand-curated ordering
# inside the symlinked set — the symlink target IS the ordering.
_N_SUBSETS: Dict[int, List[int]] = {
    1: [1],
    2: [1, 2],
    4: [1, 2, 3, 4],
    8: [1, 2, 3, 4, 5, 6, 7, 8],
}


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    """Normalise mask PNGs from the common annotation tools to fg=255, bg=0.

    Supported authoring formats (foreground = the actual mask area):
      1. RGBA with transparent background — alpha varies, fg pixels
         have alpha>0 (GIMP / Photoshop default for masks with
         "transparent bg"). Recovered via the alpha channel.
      2. RGB or RGBA-all-opaque with white background, black foreground
         — fg=(0,0,0), bg=(255,255,255). Auto-inverted by the
         majority-brightness heuristic so SegGPT sees fg=255.
      3. Single-channel grayscale 0/255 — passed through, with the same
         auto-invert applied if the foreground is dark.

    cv2.imread(IMREAD_GRAYSCALE) silently dropped alpha and returned
    all-zero for case 1, feeding 8 black prompt masks to SegGPT (96/96
    empty predictions in phase0_runs/2026-04-30_1632_*); IMREAD_UNCHANGED
    preserves the source channels so we can pick the right signal.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"mask: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if int(alpha.min()) != int(alpha.max()):
            return alpha
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.mean() > 127:
        img = 255 - img
    return img


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


def _apply_overlay(
    target_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend a solid-colour tint over `target_bgr` wherever `mask_bool` is True.

    `alpha=1.0` paints opaque; `alpha=0.0` is invisible (target untouched).
    Default 0.5 keeps the underlying scene readable while making the mask
    obvious. mask is resized to target shape via nearest-neighbour first
    so binary edges stay sharp.
    """
    if mask_bool.shape != target_bgr.shape[:2]:
        mask_bool = _resize_to(
            mask_bool.astype(np.uint8), target_bgr.shape[:2]
        ).astype(bool)
    out = target_bgr.copy()
    tint = np.array(color_bgr, dtype=np.float32)
    blended = out[mask_bool].astype(np.float32) * (1.0 - alpha) + tint * alpha
    out[mask_bool] = blended.astype(np.uint8)
    return out


def _load_driver_yaml(path: Path) -> Dict:
    """Read a driver-config YAML, validate keys, return dict.

    Missing file -> empty dict (caller falls back to hardcoded defaults).
    Unknown keys raise so silent typos don't get ignored.
    Tolerates an empty file (returns {}).
    """
    if not path.exists():
        return {}
    import yaml  # heavy-ish import deferred to keep --help fast

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(
            f"{path}: driver config must be a YAML mapping (got {type(data).__name__})"
        )
    unknown = set(data) - _DRIVER_CONFIG_KEYS
    if unknown:
        raise SystemExit(
            f"{path}: unknown driver-config keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_DRIVER_CONFIG_KEYS)}"
        )
    if "overlay_color" in data:
        color = data["overlay_color"]
        if not (isinstance(color, list) and len(color) == 3):
            raise SystemExit(
                f"{path}: overlay_color must be a 3-element BGR list; got {color!r}"
            )
        data["overlay_color"] = tuple(int(c) for c in color)
    return data


def _parse_bgr(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--overlay-color expects 'B,G,R' (3 ints 0-255); got {value!r}"
        )
    try:
        b, g, r = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--overlay-color components must be ints; got {value!r}"
        ) from exc
    for c in (b, g, r):
        if not 0 <= c <= 255:
            raise argparse.ArgumentTypeError(
                f"--overlay-color components must be 0-255; got {value!r}"
            )
    return (b, g, r)


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
    parser.add_argument(
        "--driver-config",
        type=Path,
        default=_DEFAULT_DRIVER_CONFIG,
        help="YAML override file (n_values / mode / overlay_color / "
        "overlay_alpha / no_gt). CLI flags win over YAML. "
        "Default %(default)s.",
    )
    parser.add_argument("--prompts-dir", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--targets-dir", type=Path, default=_DEFAULT_TARGETS)
    parser.add_argument("--gt-dir", type=Path, default=_DEFAULT_GTS)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output dir basename. Default: <YYYY-MM-DD_HHMM>.",
    )
    parser.add_argument(
        "--run-dir-base",
        type=Path,
        default=_REPO / "output",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--mode", choices=("instance", "semantic"), default="instance")
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip mIoU (visual inspection / latency only).",
    )
    parser.add_argument(
        "--overlay-color",
        type=_parse_bgr,
        default=(0, 0, 0),
        metavar="B,G,R",
        help="BGR tint blended over the predicted mask in overlay.png. "
        "Three 0-255 ints comma-separated. Default %(default)s (black).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Overlay blend factor: 1.0 = opaque tint, 0.0 = invisible. "
        "Default %(default)s.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # Two-pass parse so CLI > YAML > hardcoded defaults.
    # Pass 1: peek --driver-config without committing to other defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--driver-config", type=Path, default=_DEFAULT_DRIVER_CONFIG)
    pre_args, _ = pre.parse_known_args(argv)

    parser = _build_parser()
    yaml_overrides = _load_driver_yaml(pre_args.driver_config)
    if yaml_overrides:
        parser.set_defaults(**yaml_overrides)

    args = parser.parse_args(argv)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = args.run_dir_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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
        "no_gt": bool(args.no_gt),
        "overlay_color_bgr": list(args.overlay_color),
        "overlay_alpha": args.overlay_alpha,
    }

    print(f"[driver] loading model {args.model}", file=sys.stderr)
    t_load = time.perf_counter()
    from seggpt.api import SegGPTBackend  # heavy import deferred

    backend = SegGPTBackend(model_path=args.model, config_path=args.config)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    print(f"[driver] model loaded in {load_ms:.1f} ms", file=sys.stderr)
    meta["model_load_ms"] = round(load_ms, 1)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    image_dir = run_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for tgt_path in targets:
        target_img = _read_rgb(tgt_path)
        target_bgr = cv2.imread(str(tgt_path))
        cv2.imwrite(str(image_dir / f"{tgt_path.stem}.png"), target_bgr)

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

            mask_bool = pred_first.astype(bool)
            overlay = _apply_overlay(
                target_bgr, mask_bool, args.overlay_color, args.overlay_alpha
            )
            cv2.imwrite(str(image_dir / f"{tgt_path.stem}_N{n}.png"), overlay)

            if gt_mask is not None:
                pred_for_iou = _resize_to(pred_first, gt_mask.shape)
                row["miou"] = round(_miou(pred_for_iou, gt_mask), 4)

            rows.append(row)

    fieldnames = ["target", "latency_ms", "gpu_mem_mb", "mask_positive_pixels"]
    if not args.no_gt:
        fieldnames.append("miou")
    # Per-N CSVs only — keeps each N's data isolated for side-by-side
    # comparison without filtering. The "N" column is dropped because each
    # file's name already carries it.
    for n in args.n_values:
        rows_n = [
            {k: v for k, v in r.items() if k != "N"}
            for r in rows
            if r["N"] == n
        ]
        with (run_dir / f"per_image_N{n}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_n)

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
        lines.append("## Gate (main test-flow §7)")
        lines.append("")
        lines.append("| condition | value | pass |")
        lines.append("|---|---|---|")
        for label, value, thresh, op in gate:
            ok = (value >= thresh) if op == ">=" else (value <= thresh)
            lines.append(f"| {label} | {value:.3f} | {'yes' if ok else 'no'} |")
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "- Per target under `image/`: `<target_stem>.png` (raw, once) plus one "
        "`<target_stem>_N<n>.png` overlay per N value for side-by-side compare."
    )
    lines.append(
        "- Per-image numbers in `per_image_N<n>.csv`; rolled up in `n_sweep.csv` "
        "and `stats.json` under the `N=<n>` key."
    )
    lines.append(
        "- Switch prompt set: `ln -sfn <new_prompt_dir> data/phase_0_test/prompt`, "
        "then re-run the driver. Existing output stays in `output/<timestamp>/` and "
        "is not overwritten."
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
