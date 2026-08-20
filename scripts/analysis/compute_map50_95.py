#!/usr/bin/env python3
"""Compute mAP50-95 for all baselines using the KITTI evaluator.

mAP50-95 is the average of mAP computed at IoU thresholds
from 0.50 to 0.95 in steps of 0.05 (10 thresholds total),
following the COCO convention.

Usage:
    cd Fish-Monitoring
    python scripts/analysis/compute_map50_95.py \
        --pred-cache results/eval_predictions \
        --data       data/WIO-ReefFish/data.yaml \
        --split      test \
        --out-dir    results/analysis
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

from fish_monitoring.eval.kitti_eval import evaluate_kitti_style


IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95

BASELINES = [
    "rtdetr", "yolo11", "yolo26", "yolov8", "yolo-world",
    "retinanet", "dinov2", "faster-rcnn", "grounding-dino",
]

BASELINE_DISPLAY = {
    "rtdetr": "RT-DETR",
    "yolo11": "YOLO11",
    "yolo26": "YOLO26",
    "yolov8": "YOLOv8",
    "yolo-world": "YOLO-World",
    "retinanet": "RetinaNet",
    "dinov2": "DINOv2",
    "faster-rcnn": "Faster R-CNN",
    "grounding-dino": "G. DINO",
}


def _remap_labels_to_class0(src_dir: Path, dst_dir: Path) -> Path:
    """Copy label files, remapping all class IDs to 0 (class-agnostic)."""
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


def _prepare_agnostic_gt(dataset_dir: Path, split: str, tmp_root: Path) -> Path:
    """Create a temp dataset dir with GT labels remapped to cls=0."""
    out_split = tmp_root / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Symlink images
    images_link = out_split / "images"
    if not images_link.exists():
        images_link.symlink_to(dataset_dir / split / "images")

    # Remap GT labels
    _remap_labels_to_class0(
        dataset_dir / split / "labels",
        out_split / "labels",
    )
    return tmp_root


def compute_map50_95(
    data_yaml: Path,
    split: str,
    pred_labels_dir: Path,
    *,
    conf_th: float = 0.0,
    agnostic: bool = False,
) -> tuple[float, float, list[float], list[float]]:
    """Return (mAP50-95_11pt, mAP50-95_continuous, per_iou_11pt, per_iou_cont)."""
    dataset_dir = data_yaml.parent

    if agnostic:
        # Remap GT labels to class 0 for class-agnostic evaluation
        tmp_root = dataset_dir / "_tmp_agnostic_gt"
        dataset_dir = _prepare_agnostic_gt(dataset_dir, split, tmp_root)

    # Only evaluate at "Hard" (min_h=1) to get overall metrics
    difficulties = [("Overall", 1)]

    maps_11 = []
    maps_cont = []

    for iou in IOU_THRESHOLDS:
        results = evaluate_kitti_style(
            dataset_dir=dataset_dir,
            split=split,
            pred_labels_dir=pred_labels_dir,
            conf_th=conf_th,
            iou_th=float(iou),
            difficulties=difficulties,
        )
        # results is a list of DifficultyResult; we want the first (Overall)
        dr = results[0]
        maps_11.append(dr.map_11)
        maps_cont.append(dr.map_continuous)

    return float(np.mean(maps_11)), float(np.mean(maps_cont)), maps_11, maps_cont


def main():
    parser = argparse.ArgumentParser(description="Compute mAP50-95 for all baselines")
    parser.add_argument("--pred-cache", type=Path, required=True,
                        help="Root of cached prediction dirs")
    parser.add_argument("--data", type=Path, required=True,
                        help="data.yaml path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_aware = []
    rows_agnostic = []

    for baseline in BASELINES:
        print(f"\n{'='*60}")
        print(f"  {BASELINE_DISPLAY[baseline]}")
        print(f"{'='*60}")

        # Class-aware
        aware_dir = args.pred_cache / baseline / args.split / "labels"
        if aware_dir.exists():
            print(f"  [aware] evaluating at 10 IoU thresholds...")
            m11, mcont, per_11, per_cont = compute_map50_95(
                args.data, args.split, aware_dir, agnostic=False
            )
            print(f"  [aware] mAP50-95 (11pt): {m11:.4f}  |  mAP50-95 (cont): {mcont:.4f}")
            rows_aware.append({
                "baseline": baseline,
                "display": BASELINE_DISPLAY[baseline],
                "mAP50_95_11pt": f"{m11:.4f}",
                "mAP50_95_cont": f"{mcont:.4f}",
                **{f"mAP_iou{iou:.2f}": f"{v:.4f}" for iou, v in zip(IOU_THRESHOLDS, per_11)},
            })
        else:
            print(f"  [aware] labels dir not found: {aware_dir}")
            rows_aware.append({
                "baseline": baseline,
                "display": BASELINE_DISPLAY[baseline],
                "mAP50_95_11pt": "",
                "mAP50_95_cont": "",
            })

        # Class-agnostic
        agnostic_dir = args.pred_cache / baseline / args.split / "labels_agnostic"
        if agnostic_dir.exists():
            print(f"  [agnostic] evaluating at 10 IoU thresholds...")
            m11, mcont, per_11, per_cont = compute_map50_95(
                args.data, args.split, agnostic_dir, agnostic=True
            )
            print(f"  [agnostic] mAP50-95 (11pt): {m11:.4f}  |  mAP50-95 (cont): {mcont:.4f}")
            rows_agnostic.append({
                "baseline": baseline,
                "display": BASELINE_DISPLAY[baseline],
                "mAP50_95_11pt": f"{m11:.4f}",
                "mAP50_95_cont": f"{mcont:.4f}",
                **{f"mAP_iou{iou:.2f}": f"{v:.4f}" for iou, v in zip(IOU_THRESHOLDS, per_11)},
            })
        else:
            print(f"  [agnostic] labels dir not found: {agnostic_dir}")
            rows_agnostic.append({
                "baseline": baseline,
                "display": BASELINE_DISPLAY[baseline],
                "mAP50_95_11pt": "",
                "mAP50_95_cont": "",
            })

    # Write CSVs
    for label, rows in [("aware", rows_aware), ("agnostic", rows_agnostic)]:
        out_csv = args.out_dir / f"map50_95_{label}.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSaved: {out_csv}")

    # Print summary table
    print("\n\n=== SUMMARY (mAP50-95, 11-point) ===")
    print(f"{'Baseline':<16} {'Aware':>10} {'Agnostic':>10}")
    print("-" * 40)
    for ra, rag in zip(rows_aware, rows_agnostic):
        a_val = ra.get("mAP50_95_11pt", "—")
        ag_val = rag.get("mAP50_95_11pt", "—")
        print(f"{ra['display']:<16} {a_val:>10} {ag_val:>10}")


if __name__ == "__main__":
    main()
