#!/usr/bin/env python3
"""Run all analysis experiments using cached predictions.

This script runs four analyses on pre-computed prediction files
(YOLO-format TXTs cached by the evaluation pipeline):

  1. Cross-site generalization — P/R/F1 by collection site
  2. Small-object analysis    — P/R/F1 by COCO-style size bucket
  3. Visibility-conditioned   — P/R/F1 by GT box visibility (good/medium/poor)
  4. Ensemble WBF             — Weighted Box Fusion across all models

All analyses read predictions from
``results/eval_predictions/<baseline>/<split>/labels/``
and do NOT require GPU inference.

For the underwater enhancement analysis (which requires re-running inference
on enhanced images), use ``scripts/analysis/underwater_enhancement.py``
separately.

Usage:
    python scripts/analysis/run_all_cached.py \\
        --pred-cache results/eval_predictions \\
        --data       data/WIO-ReefFish/data.yaml \\
        --split      test \\
        --out-dir    results/analysis
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

from fish_monitoring.baselines.base_detector import (
    iter_split_images,
    read_yolo_labels,
    resolve_split_dirs,
)
from fish_monitoring.eval.diagnose import Gt, _load_image_size, _match_predictions
from fish_monitoring.core.inference import Pred

# bbox_attributes helpers (for visibility analysis)
from fish_monitoring.data_utils.bbox_attributes import (
    compute_box_features,
    clamp_xyxy,
    crop_region,
    load_bgr_image,
    quantize_visibility,
    read_yolo_boxes,
    yolo_xywhn_to_xyxy_px,
)


# =====================================================================
#  Shared helpers
# =====================================================================

def _remap_cls_zero(obj):
    """Remap all class IDs to 0 for class-agnostic evaluation.

    Works with frozen dataclasses (Gt, Pred) by creating new instances.
    """
    return type(obj)(**{
        f.name: (np.zeros_like(getattr(obj, f.name))
                 if f.name == "cls"
                 else getattr(obj, f.name))
        for f in obj.__dataclass_fields__.values()
    })


def load_cached_predictions(
    pred_dir: Path, img_stem: str, im_w: int, im_h: int,
) -> Pred:
    """Load YOLO-format prediction TXT as a Pred (absolute-pixel xyxy)."""
    pred_path = pred_dir / f"{img_stem}.txt"
    if not pred_path.exists():
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    boxes, confs, classes = [], [], []
    for line in pred_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            c = float(parts[5]) if len(parts) >= 6 else 1.0
        except ValueError:
            continue

        cx, cy, bw, bh = x * im_w, y * im_h, w * im_w, h * im_h
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        confs.append(c)
        classes.append(cls_id)

    if not boxes:
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    return Pred(
        xyxy=np.array(boxes, dtype=np.float32),
        conf=np.array(confs, dtype=np.float32),
        cls=np.array(classes, dtype=np.int64),
    )


def discover_baselines(
    pred_cache: Path, split: str,
) -> list[tuple[str, Path]]:
    """Discover baselines from cached prediction directories.

    Returns list of (baseline_name, labels_dir) tuples.
    """
    baselines = []
    for d in sorted(pred_cache.iterdir()):
        if not d.is_dir():
            continue
        lbl_dir = d / split / "labels"
        if lbl_dir.is_dir() and any(lbl_dir.glob("*.txt")):
            baselines.append((d.name, lbl_dir))
    return baselines


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Saved {path}")


# =====================================================================
#  1. Cross-site generalization
# =====================================================================

_SITE_RE = re.compile(r"^(.+?)_frame_", re.IGNORECASE)
_SITE_FALLBACK_RE = re.compile(r"^([A-Za-z]+)")


def extract_site(stem: str) -> str:
    """Extract site identifier from an image filename stem."""
    m = _SITE_RE.match(stem)
    if m:
        return m.group(1)
    m = _SITE_FALLBACK_RE.match(stem)
    if m:
        return m.group(1)
    return "unknown"


def analyse_cross_site(
    data_yaml: Path,
    split: str,
    pred_dir: Path,
    iou_th: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, dict[str, float]]:
    """Per-site P/R/F1 using cached predictions."""
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)

    site_tp: dict[str, int] = defaultdict(int)
    site_fp: dict[str, int] = defaultdict(int)
    site_fn: dict[str, int] = defaultdict(int)
    site_n_img: dict[str, int] = defaultdict(int)

    for img_path in image_paths:
        site = extract_site(img_path.stem)
        site_n_img[site] += 1

        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)
        gt = Gt(xyxy=gt_xyxy, cls=gt_cls)

        pred = load_cached_predictions(pred_dir, img_path.stem, w, h)

        if class_agnostic:
            gt = _remap_cls_zero(gt)
            pred = _remap_cls_zero(pred)

        tp, fp, fn, _ = _match_predictions(gt, pred, iou_th=iou_th)
        site_tp[site] += tp
        site_fp[site] += fp
        site_fn[site] += fn

    results = {}
    for site in sorted(site_tp.keys() | site_fn.keys()):
        tp = site_tp[site]
        fp = site_fp[site]
        fn = site_fn[site]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[site] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_images": site_n_img[site],
        }

    return results


# =====================================================================
#  2. Small-object analysis
# =====================================================================

# Data-driven area terciles — overridden at runtime by _compute_area_terciles()
SMALL_MAX = 2854   # default placeholder, overridden per-dataset
MEDIUM_MAX = 8324  # default placeholder, overridden per-dataset


def _box_area(xyxy: np.ndarray) -> np.ndarray:
    return (
        np.maximum(0, xyxy[:, 2] - xyxy[:, 0])
        * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    )


def _compute_area_terciles(data_yaml: Path, split: str) -> tuple[float, float]:
    """Scan GT boxes and return (p20, p55) area thresholds.

    Target distribution: Small 20% / Medium 35% / Large 45%.
    """
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)
    all_areas: list[float] = []
    for img_path in image_paths:
        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, _ = read_yolo_labels(label_path, im_w=w, im_h=h)
        if gt_xyxy.shape[0] > 0:
            areas = _box_area(gt_xyxy)
            all_areas.extend(areas.tolist())
    if not all_areas:
        return float(SMALL_MAX), float(MEDIUM_MAX)
    a = np.array(all_areas, dtype=np.float32)
    return float(np.quantile(a, 0.20)), float(np.quantile(a, 0.55))


def analyse_by_size(
    data_yaml: Path,
    split: str,
    pred_dir: Path,
    iou_th: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, dict[str, float]]:
    """Per-size-bucket P/R/F1 using cached predictions.

    Uses data-driven area terciles so each bucket contains ~33% of GT.
    """
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)

    # Compute data-driven thresholds (area terciles)
    small_max, medium_max = _compute_area_terciles(data_yaml, split)
    print(f"    Size thresholds (area terciles): small < {small_max:.0f}px²"
          f" ≤ medium < {medium_max:.0f}px² ≤ large")

    bucket_tp = {"small": 0, "medium": 0, "large": 0}
    bucket_fp = {"small": 0, "medium": 0, "large": 0}
    bucket_fn = {"small": 0, "medium": 0, "large": 0}
    bucket_n_gt = {"small": 0, "medium": 0, "large": 0}

    # Re-fetch image paths (iterator may be consumed)
    image_paths = iter_split_images(data_yaml, split)

    for img_path in image_paths:
        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)

        pred = load_cached_predictions(pred_dir, img_path.stem, w, h)

        if gt_xyxy.shape[0] == 0 and pred.xyxy.shape[0] == 0:
            continue

        gt_areas = _box_area(gt_xyxy) if gt_xyxy.shape[0] > 0 else np.array([])

        for bucket in ("small", "medium", "large"):
            if bucket == "small":
                gt_mask = gt_areas < small_max
            elif bucket == "medium":
                gt_mask = (gt_areas >= small_max) & (gt_areas < medium_max)
            else:
                gt_mask = gt_areas >= medium_max

            b_gt_xyxy = (
                gt_xyxy[gt_mask]
                if gt_xyxy.shape[0] > 0
                else np.zeros((0, 4), dtype=np.float32)
            )
            b_gt_cls = (
                gt_cls[gt_mask]
                if gt_cls.shape[0] > 0
                else np.zeros((0,), dtype=np.int64)
            )
            bucket_n_gt[bucket] += int(b_gt_xyxy.shape[0])

            gt_b = Gt(xyxy=b_gt_xyxy, cls=b_gt_cls)

            # Filter predictions by their own area to the same bucket
            if pred.xyxy.shape[0] > 0:
                pred_areas = _box_area(pred.xyxy)
                if bucket == "small":
                    pred_mask = pred_areas < small_max
                elif bucket == "medium":
                    pred_mask = (pred_areas >= small_max) & (pred_areas < medium_max)
                else:
                    pred_mask = pred_areas >= medium_max

                p_b = Pred(
                    xyxy=pred.xyxy[pred_mask],
                    conf=pred.conf[pred_mask],
                    cls=pred.cls[pred_mask],
                )
            else:
                p_b = pred

            if class_agnostic:
                gt_b = _remap_cls_zero(gt_b)
                p_b = _remap_cls_zero(p_b)

            tp, fp, fn, _ = _match_predictions(gt_b, p_b, iou_th=iou_th)
            bucket_tp[bucket] += tp
            bucket_fp[bucket] += fp
            bucket_fn[bucket] += fn

    results = {}
    for bucket in ("small", "medium", "large"):
        tp, fp, fn = bucket_tp[bucket], bucket_fp[bucket], bucket_fn[bucket]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[bucket] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gt": bucket_n_gt[bucket],
        }

    return results


# =====================================================================
#  3. Visibility-conditioned evaluation
# =====================================================================

DEFAULT_DARK_V_TH = 90.0
DEFAULT_LOW_CONTRAST_TH = 25.0


def _compute_gt_brightness_scores(
    label_path: Path,
    image_path: Path,
) -> list[float]:
    """Compute mean brightness (HSV V channel) per GT box."""
    yolo_boxes = read_yolo_boxes(label_path)
    if not yolo_boxes:
        return []

    bgr = load_bgr_image(image_path)
    im_h, im_w = bgr.shape[:2]

    scores = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = yolo_xywhn_to_xyxy_px(box, im_w=im_w, im_h=im_h)
        x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w=im_w, h=im_h)
        crop = crop_region(bgr, x1=x1, y1=y1, x2=x2, y2=y2)
        feat = compute_box_features(crop)
        scores.append(feat.mean_v)
    return scores


def _compute_brightness_quartiles(
    data_yaml: Path, split: str,
) -> tuple[float, float]:
    """Scan all GT boxes and return (p25, p75) brightness thresholds."""
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)
    all_scores: list[float] = []
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        scores = _compute_gt_brightness_scores(label_path, img_path)
        all_scores.extend(scores)
    if not all_scores:
        return 0.0, 0.0
    a = np.array(all_scores, dtype=np.float32)
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def analyse_by_visibility(
    data_yaml: Path,
    split: str,
    pred_dir: Path,
    iou_th: float = 0.5,
    dark_v_th: float = DEFAULT_DARK_V_TH,
    low_contrast_th: float = DEFAULT_LOW_CONTRAST_TH,
    class_agnostic: bool = False,
) -> dict[str, dict[str, float]]:
    """Per-brightness-quartile P/R/F1 using cached predictions.

    Uses data-driven percentile thresholds (25th/75th) on mean brightness
    (HSV V channel) to create dark/moderate/bright buckets (~25/50/25).
    """
    _, labels_dir = resolve_split_dirs(data_yaml, split)

    # Compute data-driven quartile thresholds
    t_low, t_high = _compute_brightness_quartiles(data_yaml, split)
    print(f"    Brightness quartiles: dark < {t_low:.0f} ≤ moderate < {t_high:.0f} ≤ bright")

    image_paths = iter_split_images(data_yaml, split)

    bucket_tp = {"dark": 0, "moderate": 0, "bright": 0}
    bucket_fn = {"dark": 0, "moderate": 0, "bright": 0}
    bucket_n_gt = {"dark": 0, "moderate": 0, "bright": 0}

    for img_path in image_paths:
        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)

        scores = _compute_gt_brightness_scores(label_path, img_path)

        pred = load_cached_predictions(pred_dir, img_path.stem, w, h)

        if gt_xyxy.shape[0] == 0 and pred.xyxy.shape[0] == 0:
            continue

        for bucket in ("dark", "moderate", "bright"):
            if gt_xyxy.shape[0] > 0 and scores:
                s = np.array(scores, dtype=np.float32)
                if bucket == "dark":
                    mask = s < t_low
                elif bucket == "moderate":
                    mask = (s >= t_low) & (s < t_high)
                else:
                    mask = s >= t_high
                b_gt_xyxy = gt_xyxy[mask]
                b_gt_cls = gt_cls[mask]
            else:
                b_gt_xyxy = np.zeros((0, 4), dtype=np.float32)
                b_gt_cls = np.zeros((0,), dtype=np.int64)

            bucket_n_gt[bucket] += int(b_gt_xyxy.shape[0])

            gt_b = Gt(xyxy=b_gt_xyxy, cls=b_gt_cls)
            pred_b = pred
            if class_agnostic:
                gt_b = _remap_cls_zero(gt_b)
                pred_b = _remap_cls_zero(pred)
            tp, fp, fn, _ = _match_predictions(gt_b, pred_b, iou_th=iou_th)
            bucket_tp[bucket] += tp
            bucket_fn[bucket] += fn

    results = {}
    for bucket in ("dark", "moderate", "bright"):
        tp = bucket_tp[bucket]
        fn = bucket_fn[bucket]
        n_gt = bucket_n_gt[bucket]
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_est = max(0, fn)
        prec = tp / (tp + fp_est) if (tp + fp_est) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[bucket] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fn": fn,
            "n_gt": n_gt,
        }

    return results


# =====================================================================
#  4. Ensemble WBF
# =====================================================================

def weighted_box_fusion(
    predictions: list[Pred],
    im_w: int,
    im_h: int,
    iou_th: float = 0.55,
    conf_th: float = 0.01,
) -> Pred:
    """Weighted Box Fusion for an ensemble of detectors on one image."""
    n_models = len(predictions)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for pred in predictions:
        if pred.xyxy.shape[0] == 0:
            continue
        boxes_norm = pred.xyxy.copy()
        boxes_norm[:, [0, 2]] /= im_w
        boxes_norm[:, [1, 3]] /= im_h
        boxes_norm = np.clip(boxes_norm, 0, 1)

        all_boxes.append(boxes_norm)
        all_scores.append(pred.conf.copy())
        all_labels.append(pred.cls.copy())

    if not all_boxes:
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]

    used = np.zeros(len(boxes), dtype=bool)
    fused_boxes, fused_scores, fused_labels = [], [], []

    for i in range(len(boxes)):
        if used[i]:
            continue

        cluster_boxes = [boxes[i]]
        cluster_scores = [scores[i]]
        cluster_label = labels[i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j] or labels[j] != cluster_label:
                continue
            cluster_box = np.average(cluster_boxes, weights=cluster_scores, axis=0)
            iou = _compute_iou(cluster_box, boxes[j])
            if iou >= iou_th:
                cluster_boxes.append(boxes[j])
                cluster_scores.append(scores[j])
                used[j] = True

        cluster_boxes_arr = np.array(cluster_boxes)
        cluster_scores_arr = np.array(cluster_scores)

        fused_box = np.average(cluster_boxes_arr, weights=cluster_scores_arr, axis=0)
        n_contributing = len(cluster_scores_arr)
        avg_score = cluster_scores_arr.mean()
        fused_conf = avg_score * min(n_contributing, n_models) / n_models

        fused_boxes.append(fused_box)
        fused_scores.append(fused_conf)
        fused_labels.append(cluster_label)

    if not fused_boxes:
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    out_boxes = np.array(fused_boxes, dtype=np.float32)
    out_scores = np.array(fused_scores, dtype=np.float32)
    out_labels = np.array(fused_labels, dtype=np.int64)

    keep = out_scores >= conf_th
    out_boxes = out_boxes[keep]
    out_scores = out_scores[keep]
    out_labels = out_labels[keep]

    # De-normalise
    if out_boxes.shape[0] > 0:
        out_boxes[:, [0, 2]] *= im_w
        out_boxes[:, [1, 3]] *= im_h

    return Pred(xyxy=out_boxes, conf=out_scores, cls=out_labels)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def run_ensemble_wbf(
    data_yaml: Path,
    split: str,
    baselines: list[tuple[str, Path]],
    iou_th: float = 0.5,
    wbf_iou: float = 0.55,
    conf_th: float = 0.25,
    class_agnostic: bool = False,
) -> tuple[dict[str, dict], dict]:
    """Run WBF ensemble and return (per_model_metrics, ensemble_metrics)."""
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)

    model_tp = {bl: 0 for bl, _ in baselines}
    model_fp = {bl: 0 for bl, _ in baselines}
    model_fn = {bl: 0 for bl, _ in baselines}

    ens_tp = ens_fp = ens_fn = 0

    for img_path in image_paths:
        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)
        gt = Gt(xyxy=gt_xyxy, cls=gt_cls)
        if class_agnostic:
            gt = _remap_cls_zero(gt)

        per_model_preds = []
        for bl_name, pred_dir in baselines:
            pred = load_cached_predictions(pred_dir, img_path.stem, w, h)
            if class_agnostic:
                pred = _remap_cls_zero(pred)
            per_model_preds.append(pred)

            tp, fp, fn, _ = _match_predictions(gt, pred, iou_th=iou_th)
            model_tp[bl_name] += tp
            model_fp[bl_name] += fp
            model_fn[bl_name] += fn

        fused = weighted_box_fusion(
            per_model_preds,
            im_w=w,
            im_h=h,
            iou_th=wbf_iou,
            conf_th=conf_th,
        )

        tp, fp, fn, _ = _match_predictions(gt, fused, iou_th=iou_th)
        ens_tp += tp
        ens_fp += fp
        ens_fn += fn

    per_model_results = {}
    for bl_name, _ in baselines:
        tp, fp, fn = model_tp[bl_name], model_fp[bl_name], model_fn[bl_name]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_model_results[bl_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    prec = ens_tp / (ens_tp + ens_fp) if (ens_tp + ens_fp) > 0 else 0.0
    rec = ens_tp / (ens_tp + ens_fn) if (ens_tp + ens_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ensemble_results = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": ens_tp,
        "fp": ens_fp,
        "fn": ens_fn,
    }

    return per_model_results, ensemble_results


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all analysis experiments using cached predictions"
    )
    parser.add_argument(
        "--pred-cache",
        required=True,
        help="Dir with cached predictions (e.g. results/eval_predictions)",
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--dark-v-th", type=float, default=DEFAULT_DARK_V_TH)
    parser.add_argument("--low-contrast-th", type=float, default=DEFAULT_LOW_CONTRAST_TH)
    parser.add_argument(
        "--out-dir",
        default="results/analysis",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help="Specific baselines to analyse (default: all in pred-cache)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Baselines to skip (e.g. baselines with zero predictions)",
    )
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        default=False,
        help="Remap all classes to 0 (class-agnostic / fish-vs-background)",
    )
    args = parser.parse_args()

    pred_cache = Path(args.pred_cache).resolve()
    data_yaml = Path(args.data).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover baselines
    all_baselines = discover_baselines(pred_cache, args.split)
    if args.baselines:
        all_baselines = [(n, p) for n, p in all_baselines if n in args.baselines]
    if args.skip:
        all_baselines = [(n, p) for n, p in all_baselines if n not in args.skip]

    if not all_baselines:
        print("ERROR: No baselines found in", pred_cache)
        sys.exit(1)

    bl_names = [n for n, _ in all_baselines]
    t0 = time.time()

    class_agnostic = args.class_agnostic
    mode_label = "class-AGNOSTIC" if class_agnostic else "class-aware"

    print(f"\n{'='*80}")
    print(f"  Analysis Experiments — {args.split} split ({mode_label})")
    print(f"  Baselines ({len(all_baselines)}): {', '.join(bl_names)}")
    print(f"  Pred cache: {pred_cache}")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*80}")

    # ── 1. Cross-site evaluation ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  1. Cross-Site Generalization")
    print(f"{'─'*70}")

    cross_site_rows: list[dict] = []
    for bl_name, pred_dir in all_baselines:
        print(f"\n    {bl_name}")
        result = analyse_cross_site(
            data_yaml, args.split, pred_dir, iou_th=args.iou,
            class_agnostic=class_agnostic,
        )

        f1_values = []
        for site in sorted(result.keys()):
            m = result[site]
            f1_values.append(m["f1"])
            print(
                f"      {site:>30s}:  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"(imgs={m['n_images']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']})"
            )
            cross_site_rows.append({
                "baseline": bl_name,
                "site": site,
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
                "f1": f"{m['f1']:.4f}",
                "n_images": m["n_images"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            })

        if f1_values:
            arr = np.array(f1_values)
            print(
                f"\n      {'SUMMARY':>30s}:  mean_F1={arr.mean():.3f}  "
                f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}"
            )
            cross_site_rows.append({
                "baseline": bl_name,
                "site": "SUMMARY",
                "precision": "",
                "recall": "",
                "f1": f"{arr.mean():.4f} ± {arr.std():.4f}",
                "n_images": sum(result[s]["n_images"] for s in result),
                "tp": "",
                "fp": "",
                "fn": "",
            })

    _write_csv(cross_site_rows, out_dir / "cross_site_eval.csv")

    # ── 2. Small-object analysis ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  2. Small-Object Analysis")
    print(f"  COCO buckets: small < {SMALL_MAX} px²  |  medium < {MEDIUM_MAX} px²  |  large ≥ {MEDIUM_MAX} px²")
    print(f"{'─'*70}")

    small_obj_rows: list[dict] = []
    for bl_name, pred_dir in all_baselines:
        print(f"\n    {bl_name}")
        result = analyse_by_size(
            data_yaml, args.split, pred_dir, iou_th=args.iou,
            class_agnostic=class_agnostic,
        )

        for bucket in ("small", "medium", "large"):
            m = result[bucket]
            print(
                f"      {bucket:>6s}:  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"(GT={m['n_gt']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']})"
            )
            small_obj_rows.append({
                "baseline": bl_name,
                "size_bucket": bucket,
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
                "f1": f"{m['f1']:.4f}",
                "n_gt": m["n_gt"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            })

    _write_csv(small_obj_rows, out_dir / "small_object_analysis.csv")

    # ── 3. Visibility-conditioned evaluation ─────────────────────────
    print(f"\n{'─'*70}")
    print("  3. Visibility-Conditioned Evaluation")
    print(
        f"  Thresholds: dark_v < {args.dark_v_th}, "
        f"low_contrast_std_l < {args.low_contrast_th}"
    )
    print(f"{'─'*70}")

    vis_rows: list[dict] = []
    for bl_name, pred_dir in all_baselines:
        print(f"\n    {bl_name}")
        result = analyse_by_visibility(
            data_yaml,
            args.split,
            pred_dir,
            iou_th=args.iou,
            dark_v_th=args.dark_v_th,
            low_contrast_th=args.low_contrast_th,
            class_agnostic=class_agnostic,
        )

        for bucket in ("dark", "moderate", "bright"):
            m = result[bucket]
            print(
                f"      {bucket:>8s}:  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"(GT={m['n_gt']}  TP={m['tp']}  FN={m['fn']})"
            )
            vis_rows.append({
                "baseline": bl_name,
                "visibility": bucket,
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
                "f1": f"{m['f1']:.4f}",
                "n_gt": m["n_gt"],
                "tp": m["tp"],
                "fn": m["fn"],
            })

    _write_csv(vis_rows, out_dir / "visibility_eval.csv")

    # ── 4. Ensemble WBF ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  4. Ensemble WBF")
    print(f"  WBF IoU threshold: {args.wbf_iou}")

    # Filter out baselines with zero total predictions
    active_baselines = []
    for bl_name, pred_dir in all_baselines:
        n_nonempty = sum(
            1 for f in pred_dir.glob("*.txt") if f.stat().st_size > 0
        )
        if n_nonempty > 0:
            active_baselines.append((bl_name, pred_dir))
        else:
            print(f"  SKIP {bl_name} (zero predictions)")

    print(f"  Active models: {', '.join(n for n, _ in active_baselines)}")
    print(f"{'─'*70}")

    per_model, ensemble = run_ensemble_wbf(
        data_yaml,
        args.split,
        active_baselines,
        iou_th=args.iou,
        wbf_iou=args.wbf_iou,
        conf_th=args.conf,
        class_agnostic=class_agnostic,
    )

    wbf_rows: list[dict] = []
    for bl_name, _ in active_baselines:
        m = per_model[bl_name]
        print(
            f"    {bl_name:>20s}:  P={m['precision']:.3f}  "
            f"R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )
        wbf_rows.append({
            "model": bl_name,
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

    m = ensemble
    print(
        f"\n    {'ENSEMBLE (WBF)':>20s}:  P={m['precision']:.3f}  "
        f"R={m['recall']:.3f}  F1={m['f1']:.3f}"
    )
    wbf_rows.append({
        "model": "ENSEMBLE_WBF",
        "precision": f"{m['precision']:.4f}",
        "recall": f"{m['recall']:.4f}",
        "f1": f"{m['f1']:.4f}",
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
    })

    best_f1 = max(per_model[bl]["f1"] for bl, _ in active_baselines)
    delta = m["f1"] - best_f1
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
    print(f"    Δ F1 vs best individual: {arrow} {delta:+.4f}")

    _write_csv(wbf_rows, out_dir / "ensemble_wbf.csv")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  All analyses complete in {elapsed:.1f}s")
    print(f"  Results saved to: {out_dir}/")
    print(f"    • cross_site_eval.csv")
    print(f"    • small_object_analysis.csv")
    print(f"    • visibility_eval.csv")
    print(f"    • ensemble_wbf.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
