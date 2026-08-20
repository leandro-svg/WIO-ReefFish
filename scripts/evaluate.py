#!/usr/bin/env python3
"""Unified evaluation: class-aware vs class-agnostic, with difficulty levels.

Evaluates every trained baseline on a WIO-ReefFish split using two
complementary protocols:

  1. **Class-aware** (``--mode aware``)
     A prediction is a true positive only if it matches both the location
     (IoU >= 0.50) and the ground-truth fish family. mAP50 is the mean AP
     over all families in ``data.yaml``.

  2. **Class-agnostic** (``--mode agnostic``)
     All ground-truth and predicted class IDs are remapped to 0 ("fish"),
     so only localisation is scored. This gives open-vocabulary and
     segmentation models (Grounding DINO, SAM 2) a fair comparison against
     fine-tuned multi-class detectors.

Both modes are additionally broken down by KITTI-style difficulty levels
derived from ground-truth bounding-box height quartiles
(Overall / Easy / Moderate / Hard).

Weights are auto-discovered under ``--results`` as
``<baseline>*/weights/best.pt`` (Ultralytics) or ``<baseline>*/best.pt``
(torchvision-style baselines). Baselines with no weights are skipped;
``grounding-dino`` is always run zero-shot.

Outputs (written under ``--results``):
    eval_<mode>.csv             per-baseline x split x difficulty metrics
    eval_<mode>_summary.md      ranked markdown tables
    eval_predictions/           cached YOLO-format predictions, reused across modes

Examples:
    python scripts/evaluate.py --data data/WIO-ReefFish/data.yaml --results results
    python scripts/evaluate.py --data data/WIO-ReefFish/data.yaml --results results \
        --mode agnostic --splits test --baselines yolo26 rtdetr
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Runtime configuration (set by configure(), see main()) ───────────────
DATA_YAML: Path = Path()
DATASET_DIR: Path = Path()
RESULTS_DIR: Path = Path()
CLASS_NAMES: list[str] = []
EVAL_JOBS: list[tuple[str, str, float]] = []
IMGSZ = 640
IOU_TH = 0.5

# (baseline name, default confidence threshold)
_BASELINE_CONFS = [
    ("rtdetr",         0.25),
    ("yolov8",         0.25),
    ("yolo11",         0.25),
    ("yolo26",         0.25),
    ("yolo-world",     0.25),
    ("faster-rcnn",    0.25),
    ("retinanet",      0.25),
    ("dinov2",         0.25),
    ("dinov2-frcnn",   0.25),
    ("sam2",           0.25),
    ("grounding-dino", 0.30),
    ("co-detr",        0.25),
]


def _find_weights(baseline_name: str) -> str | None:
    """Auto-discover the best checkpoint for a baseline under RESULTS_DIR."""
    import glob
    for pattern in [
        str(RESULTS_DIR / baseline_name / "weights" / "best.pt"),
        str(RESULTS_DIR / f"{baseline_name}_*" / "weights" / "best.pt"),
        str(RESULTS_DIR / baseline_name / "best.pt"),
        str(RESULTS_DIR / f"{baseline_name}_*" / "best.pt"),
        str(RESULTS_DIR / baseline_name / "best.pth"),
        str(RESULTS_DIR / f"{baseline_name}_*" / "best.pth"),
    ]:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # latest by name
    return None


def configure(data: Path, results: Path, imgsz: int = 640, iou: float = 0.5) -> None:
    """Point the module at a dataset + results directory and discover weights."""
    global DATA_YAML, DATASET_DIR, RESULTS_DIR, CLASS_NAMES, EVAL_JOBS, IMGSZ, IOU_TH
    import yaml

    data = Path(data).expanduser().resolve()
    if data.is_dir():
        data = data / "data.yaml"
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    DATA_YAML = data
    DATASET_DIR = data.parent
    RESULTS_DIR = Path(results).expanduser().resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMGSZ, IOU_TH = imgsz, iou

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        CLASS_NAMES = yaml.safe_load(f)["names"]

    EVAL_JOBS = []
    for name, conf in _BASELINE_CONFS:
        if name == "grounding-dino":
            EVAL_JOBS.append((name, "auto", conf))  # zero-shot, no checkpoint
        else:
            EVAL_JOBS.append((name, _find_weights(name) or "MISSING", conf))


# ── Imports (after sys.path) ─────────────────────────────────────────────
from fish_monitoring.baselines import get_baseline
from fish_monitoring.baselines.base_detector import (
    iter_split_images,
    save_yolo_predictions,
    IMAGE_EXTS,
)
from fish_monitoring.core.inference import Pred
from fish_monitoring.eval.kitti_eval import (
    evaluate_kitti_style,
    DifficultyResult,
)
from fish_monitoring.eval.metrics import derive_difficulty_thresholds


# =====================================================================
#  Helpers
# =====================================================================

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _run_inference(
    baseline_name: str,
    model_path_str: str,
    split: str,
    conf: float,
    cache_dir: Path,
) -> Path:
    """Run inference and cache YOLO-format prediction labels.

    Returns the path to the labels/ directory.
    """
    from PIL import Image

    labels_dir = cache_dir / "labels"

    # Re-use cached predictions if they exist and are non-empty
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        n = len(list(labels_dir.glob("*.txt")))
        print(f"    Using cached predictions ({n} files) in {labels_dir}")
        return labels_dir

    labels_dir.mkdir(parents=True, exist_ok=True)

    detector = get_baseline(baseline_name)
    image_paths = iter_split_images(DATA_YAML, split)
    model_path = Path(model_path_str)

    print(f"    Inferring {len(image_paths)} images …")
    for i, img_path in enumerate(image_paths):
        pred = detector.predict(
            img_path,
            model_path=model_path,
            imgsz=IMGSZ,
            conf=conf,
            iou=IOU_TH,
            device=0,
        )
        w, h = Image.open(img_path).size
        save_yolo_predictions(pred, img_path, labels_dir, im_w=w, im_h=h)
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(image_paths)} done")

    print(f"    Saved {len(image_paths)} prediction files")
    return labels_dir


def _remap_labels_to_class0(src_dir: Path, dst_dir: Path) -> Path:
    """Copy label files, remapping all class IDs to 0.

    Used for class-agnostic evaluation: both GT and predictions are remapped
    so the matching ignores the class and only checks IoU.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for label_file in sorted(src_dir.glob("*.txt")):
        lines = label_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = "0"
                new_lines.append(" ".join(parts))
        (dst_dir / label_file.name).write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )
    return dst_dir


def _prepare_agnostic_gt(split: str, tmp_root: Path) -> Path:
    """Create a temporary dataset directory with GT labels remapped to cls=0.

    Returns a Path that looks like a dataset dir:
        <tmp_root>/<split>/images  (symlink)
        <tmp_root>/<split>/labels  (remapped copies)
    """
    out_split = tmp_root / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Symlink images
    images_link = out_split / "images"
    if not images_link.exists():
        images_link.symlink_to(DATASET_DIR / split / "images")

    # Remap GT labels
    _remap_labels_to_class0(
        DATASET_DIR / split / "labels",
        out_split / "labels",
    )
    return tmp_root


def _evaluate_split(
    baseline_name: str,
    pred_labels_dir: Path,
    split: str,
    conf: float,
    mode: Literal["aware", "agnostic"],
    with_difficulty: bool = True,
) -> list[DifficultyResult]:
    """Run KITTI-style evaluation, optionally with difficulty levels."""

    if mode == "agnostic":
        # Remap GT to class 0
        tmp_root = RESULTS_DIR / "eval_tmp" / "agnostic_gt"
        dataset_dir = _prepare_agnostic_gt(split, tmp_root)
        num_classes = 1

        # Remap predictions to class 0
        remapped_preds = pred_labels_dir.parent / "labels_agnostic"
        _remap_labels_to_class0(pred_labels_dir, remapped_preds)
        pred_labels_dir = remapped_preds
    else:
        dataset_dir = DATASET_DIR
        num_classes = len(CLASS_NAMES)

    # Derive difficulty thresholds from GT heights
    if with_difficulty:
        easy_h, moderate_h, hard_h, stats = derive_difficulty_thresholds(
            dataset_dir=dataset_dir, split=split,
        )
        print(f"    Difficulty: Easy≥{easy_h}px  Moderate≥{moderate_h}px  Hard≥{hard_h}px")

        difficulties = [
            {"name": "Overall",  "min_h": 1,          "type": "none"},
            {"name": "Easy",     "min_h": easy_h,      "type": "none"},
            {"name": "Moderate", "min_h": moderate_h,   "type": "none"},
            {"name": "Hard",     "min_h": hard_h,       "type": "none"},
        ]
    else:
        difficulties = [
            {"name": "Overall", "min_h": 1, "type": "none"},
        ]

    results = evaluate_kitti_style(
        dataset_dir=dataset_dir,
        split=split,
        pred_labels_dir=pred_labels_dir,
        conf_th=conf,
        iou_th=IOU_TH,
        difficulties=difficulties,
        num_classes=num_classes,
    )
    return results


# =====================================================================
#  Summary generation
# =====================================================================

def _write_summary_md(
    all_rows: list[dict],
    mode: str,
    out_path: Path,
) -> None:
    """Write a markdown summary of evaluation results."""
    mode_label = (
        f"Class-Aware ({len(CLASS_NAMES)} families)" if mode == "aware"
        else "Class-Agnostic (fish vs. background)"
    )

    lines = [
        f"# Evaluation Results — {mode_label}\n",
        f"> **Dataset**: `{DATA_YAML}`  ",
        f"> **IoU threshold**: {IOU_TH:.2f} | **Image size**: {IMGSZ}×{IMGSZ}  ",
        f"> **Mode**: {mode_label}  \n",
    ]

    # Group by split
    for split in ("test", "valid"):
        split_rows = [r for r in all_rows if r["split"] == split]
        if not split_rows:
            continue

        lines.append(f"\n## {split.capitalize()} Split\n")

        # Overall table (ranked by mAP50)
        overall = [r for r in split_rows if r["difficulty"] == "Overall"]
        overall.sort(key=lambda r: r["mAP50_11pt"], reverse=True)

        lines.append("### Overall (all object sizes)\n")
        lines.append("| Rank | Baseline | P | R | F1 | mAP50 | TP | FP | FN |")
        lines.append("|:----:|:---------|----:|----:|----:|------:|---:|---:|---:|")
        for i, r in enumerate(overall, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
            lines.append(
                f"| {medal} | **{r['baseline']}** | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                f"{r['mAP50_11pt']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} |"
            )

        # Per-difficulty compact table
        diff_rows = [r for r in split_rows if r["difficulty"] != "Overall"]
        if diff_rows:
            # Get baselines sorted by Overall mAP50
            baseline_order = [r["baseline"] for r in overall]
            lines.append("\n### mAP50 by Difficulty\n")
            lines.append("| Baseline | Easy | Moderate | Hard | Δ Easy→Hard |")
            lines.append("|:---------|-----:|---------:|-----:|------------:|")
            for bl in baseline_order:
                bl_rows = {r["difficulty"]: r for r in diff_rows if r["baseline"] == bl}
                easy = bl_rows.get("Easy", {}).get("mAP50_11pt", 0)
                mod = bl_rows.get("Moderate", {}).get("mAP50_11pt", 0)
                hard = bl_rows.get("Hard", {}).get("mAP50_11pt", 0)
                delta = hard - easy if easy > 0 else 0
                lines.append(f"| {bl} | {easy:.3f} | {mod:.3f} | {hard:.3f} | {delta:+.3f} |")

            lines.append("\n### F1 by Difficulty\n")
            lines.append("| Baseline | Easy | Moderate | Hard | Δ Easy→Hard |")
            lines.append("|:---------|-----:|---------:|-----:|------------:|")
            for bl in baseline_order:
                bl_rows = {r["difficulty"]: r for r in diff_rows if r["baseline"] == bl}
                easy = bl_rows.get("Easy", {}).get("f1", 0)
                mod = bl_rows.get("Moderate", {}).get("f1", 0)
                hard = bl_rows.get("Hard", {}).get("f1", 0)
                delta = hard - easy if easy > 0 else 0
                lines.append(f"| {bl} | {easy:.3f} | {mod:.3f} | {hard:.3f} | {delta:+.3f} |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary written to: {out_path}")


# =====================================================================
#  Main
# =====================================================================

def run_eval(
    mode: Literal["aware", "agnostic"],
    baselines: list[str] | None = None,
    splits: list[str] | None = None,
    with_difficulty: bool = True,
) -> None:
    """Run one evaluation mode across all baselines."""
    splits = splits or ["valid", "test"]
    mode_label = "class-aware" if mode == "aware" else "class-agnostic"

    csv_path = RESULTS_DIR / f"eval_{mode}.csv"
    md_path = RESULTS_DIR / f"eval_{mode}_summary.md"

    print("=" * 70)
    print(f"  Evaluation mode: {mode_label}")
    print(f"  Splits: {', '.join(splits)}")
    print(f"  Difficulty levels: {'Yes' if with_difficulty else 'No'}")
    print(f"  Output CSV: {csv_path}")
    print("=" * 70)

    header = [
        "baseline", "split", "difficulty", "mode",
        "precision", "recall", "f1", "tp", "fp", "fn",
        "mAP50_11pt", "mAP50_continuous",
    ]
    all_rows: list[dict] = []

    jobs = EVAL_JOBS
    if baselines:
        allowed = set(b.lower().strip() for b in baselines)
        jobs = [(n, w, c) for n, w, c in jobs if n in allowed]

    for baseline_name, weights_path, conf in jobs:
        # Check weights exist
        if weights_path != "auto" and not Path(weights_path).exists():
            print(f"\n  [SKIP] {baseline_name}: weights not found")
            continue

        for split in splits:
            print(f"\n{'─' * 60}")
            print(f"  {baseline_name} / {split} ({mode_label})")
            print(f"{'─' * 60}")
            t0 = time.time()

            # Step 1: Inference (cached across modes)
            pred_cache = RESULTS_DIR / "eval_predictions" / baseline_name / split
            pred_labels_dir = _run_inference(
                baseline_name, weights_path, split, conf, pred_cache,
            )
            t_infer = time.time() - t0

            # Step 2: KITTI evaluation
            results = _evaluate_split(
                baseline_name, pred_labels_dir, split, conf,
                mode=mode,
                with_difficulty=with_difficulty,
            )
            t_total = time.time() - t0
            print(f"    Time: inference={t_infer:.1f}s  total={t_total:.1f}s")

            for r in results:
                p = float(r.point.precision)
                rec = float(r.point.recall)
                f1 = _safe_div(2 * p * rec, p + rec)

                row = {
                    "baseline": baseline_name,
                    "split": split,
                    "difficulty": r.name,
                    "mode": mode,
                    "precision": p,
                    "recall": rec,
                    "f1": f1,
                    "tp": r.point.tp,
                    "fp": r.point.fp,
                    "fn": r.point.fn,
                    "mAP50_11pt": r.map_11,
                    "mAP50_continuous": r.map_continuous,
                }
                all_rows.append(row)
                print(
                    f"    {r.name:>10s}  P={p:.3f}  R={rec:.3f}  F1={f1:.3f}  "
                    f"mAP50={r.map_11:.3f}  TP={r.point.tp}  FP={r.point.fp}  FN={r.point.fn}"
                )

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  CSV written to: {csv_path}")

    # Write summary
    _write_summary_md(all_rows, mode, md_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified evaluation: class-aware vs class-agnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the split's data.yaml (or the directory containing it)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results directory holding trained weights; outputs are written here",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument(
        "--mode",
        choices=["aware", "agnostic", "both"],
        default="both",
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help="Subset of baselines to evaluate (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to evaluate on (default: valid test)",
    )
    parser.add_argument(
        "--no-difficulty",
        action="store_true",
        help="Skip per-difficulty breakdown (Overall only)",
    )
    args = parser.parse_args()

    configure(args.data, args.results, imgsz=args.imgsz, iou=args.iou)
    print(f"Dataset: {DATA_YAML}  ({len(CLASS_NAMES)} classes)")
    print(f"Results: {RESULTS_DIR}")

    import torch
    print(f"Python: {sys.executable}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")

    modes: list[str] = []
    if args.mode == "both":
        modes = ["aware", "agnostic"]
    else:
        modes = [args.mode]

    for mode in modes:
        run_eval(
            mode=mode,  # type: ignore[arg-type]
            baselines=args.baselines,
            splits=args.splits,
            with_difficulty=not args.no_difficulty,
        )

    # Clean up temp dirs
    tmp_dir = RESULTS_DIR / "eval_tmp"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n✅ All evaluations complete.")


if __name__ == "__main__":
    main()
