"""Markdown performance report generation.

Builds human-readable Markdown reports from KITTI-style evaluation
results, including per-class and per-difficulty breakdowns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from fish_monitoring.eval.kitti_eval import DifficultyResult, evaluate_kitti_style


@dataclass(frozen=True)
class ReportConfig:
    dataset_dir: Path
    split: str
    pred_dir: Path
    out_md: Path

    conf_th: float = 0.52
    iou_th: float = 0.5

    # KITTI-like difficulty buckets, but using ONLY min bbox height (px)
    # since we don't have occlusion/truncation labels.
    easy_min_h: int | None = None
    moderate_min_h: int | None = None
    hard_min_h: int | None = None

    num_classes: int | None = None

    # New difficulty dimensions (percentile thresholds in [0,1])
    # If provided, report will include additional difficulty slices:
    # - size / visibility / background / combined
    # using these thresholds.
    size_pct: float | None = None
    visibility_pct: float | None = None
    background_pct: float | None = None


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _difficulty_section(r: DifficultyResult, class_names: list[str] | None) -> str:
    lines: list[str] = []
    lines.append(f"## {r.name}\n")
    lines.append(f"- IoU threshold: `{r.iou_th:.2f}`\n")
    lines.append(f"- Min bbox height: `{r.min_height_px}` px\n")
    if getattr(r, "size_pct", None) is not None:
        lines.append(f"- Size threshold (pct): `{float(r.size_pct):.2f}`\n")
    if getattr(r, "visibility_pct", None) is not None:
        lines.append(f"- Visibility threshold (pct): `{float(r.visibility_pct):.2f}`\n")
    if getattr(r, "background_pct", None) is not None:
        lines.append(f"- Background threshold (pct): `{float(r.background_pct):.2f}`\n")
    lines.append("\n")

    lines.append("### Point Metrics (at chosen confidence)\n")
    lines.append(f"- TP/FP/FN: `{r.point.tp}` / `{r.point.fp}` / `{r.point.fn}`\n")
    lines.append(f"- Precision: `{_fmt(r.point.precision)}`\n")
    lines.append(f"- Recall: `{_fmt(r.point.recall)}`\n")
    lines.append("\n")

    lines.append("### mAP\n")
    lines.append(f"- mAP (VOC 11-point): `{_fmt(r.map_11)}`\n")
    lines.append(f"- mAP (continuous): `{_fmt(r.map_continuous)}`\n")
    lines.append("\n")

    lines.append("### Per-class (Point Metrics)\n")
    lines.append("| class | tp | fp | fn | precision | recall | ap11 | ap_cont |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")

    for cls_id in sorted(r.pr_curves.keys()):
        curve = r.pr_curves[cls_id]
        pm = r.per_class.get(cls_id)
        if pm is None:
            tp = fp = fn = 0
            prec = rec = 0.0
        else:
            tp, fp, fn = pm.tp, pm.fp, pm.fn
            prec, rec = pm.precision, pm.recall

        name = str(cls_id)
        if class_names is not None and 0 <= cls_id < len(class_names):
            name = class_names[cls_id]

        # Show AP even if n_gt=0 (will be 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(tp),
                    str(fp),
                    str(fn),
                    _fmt(prec),
                    _fmt(rec),
                    _fmt(curve.ap_11),
                    _fmt(curve.ap_continuous),
                ]
            )
            + " |\n"
        )

    lines.append("\n")
    return "".join(lines)


def _difficulty_summary_table(results: list[DifficultyResult]) -> str:
    lines: list[str] = []
    lines.append("### Summary (Precision/Recall/mAP)\n")
    lines.append("| difficulty | min_h_px | tp | fp | fn | precision | recall | mAP11 | mAP_cont |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    str(int(r.min_height_px)),
                    str(int(r.point.tp)),
                    str(int(r.point.fp)),
                    str(int(r.point.fn)),
                    _fmt(r.point.precision),
                    _fmt(r.point.recall),
                    _fmt(r.map_11),
                    _fmt(r.map_continuous),
                ]
            )
            + " |\n"
        )
    lines.append("\n")
    return "".join(lines)


def write_performance_report(cfg: ReportConfig, *, class_names: list[str] | None = None) -> Path:
    pred_labels_dir = cfg.pred_dir
    if pred_labels_dir.is_dir() and (pred_labels_dir / "labels").is_dir():
        # allow passing the parent infer output dir
        pred_labels_dir = pred_labels_dir / "labels"

    # Derive KITTI-like min-height thresholds from dataset statistics if not provided.
    # We use GT bbox height percentiles: Easy=p75, Moderate=p50, Hard=p25.
    from fish_monitoring.eval.kitti_eval import _heights_xyxy, read_yolo_labels  # type: ignore

    def _load_image_size(image_path: Path) -> tuple[int, int]:
        from PIL import Image

        im = Image.open(image_path)
        return im.size

    images_dir = cfg.dataset_dir / cfg.split / "images"
    gt_labels_dir = cfg.dataset_dir / cfg.split / "labels"
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    heights: list[float] = []
    for image_path in images:
        im_w, im_h = _load_image_size(image_path)
        gt_path = gt_labels_dir / f"{image_path.stem}.txt"
        gt = read_yolo_labels(gt_path, im_w=im_w, im_h=im_h, has_conf=False)
        heights.extend(_heights_xyxy(gt.xyxy).astype(float).tolist())

    h = np.array(heights, dtype=np.float32)
    if h.size:
        p25 = int(round(float(np.quantile(h, 0.25))))
        p50 = int(round(float(np.quantile(h, 0.50))))
        p75 = int(round(float(np.quantile(h, 0.75))))
        hmin = float(h.min(initial=0.0))
        hmax = float(h.max(initial=0.0))
    else:
        p25, p50, p75 = 15, 25, 40
        hmin, hmax = 0.0, 0.0

    easy_h = int(cfg.easy_min_h) if cfg.easy_min_h is not None else max(1, p75)
    mod_h = int(cfg.moderate_min_h) if cfg.moderate_min_h is not None else max(1, min(easy_h, p50))
    hard_h = int(cfg.hard_min_h) if cfg.hard_min_h is not None else max(1, min(mod_h, p25))

    # Legacy height-bucket difficulties
    diffs_legacy = [("Overall", 1), ("Easy", easy_h), ("Moderate", mod_h), ("Hard", hard_h)]

    results_legacy = evaluate_kitti_style(
        dataset_dir=cfg.dataset_dir,
        split=cfg.split,
        pred_labels_dir=pred_labels_dir,
        conf_th=float(cfg.conf_th),
        iou_th=float(cfg.iou_th),
        difficulties=diffs_legacy,
        num_classes=cfg.num_classes,
    )

    # New 4-type difficulty evaluation (optional)
    results_attr: list[DifficultyResult] | None = None
    if (cfg.size_pct is not None) or (cfg.visibility_pct is not None) or (cfg.background_pct is not None):
        diffs_attr = [
            {"name": "Size", "type": "size", "min_size_pct": (float(cfg.size_pct) if cfg.size_pct is not None else 0.5)},
            {
                "name": "Visibility",
                "type": "visibility",
                "min_visibility_pct": (float(cfg.visibility_pct) if cfg.visibility_pct is not None else 0.5),
            },
            {
                "name": "Background",
                "type": "background",
                "min_background_pct": (float(cfg.background_pct) if cfg.background_pct is not None else 0.5),
            },
            {
                "name": "Combined",
                "type": "combined",
                "min_size_pct": (float(cfg.size_pct) if cfg.size_pct is not None else 0.5),
                "min_visibility_pct": (float(cfg.visibility_pct) if cfg.visibility_pct is not None else 0.5),
                "min_background_pct": (float(cfg.background_pct) if cfg.background_pct is not None else 0.5),
            },
        ]

        results_attr = evaluate_kitti_style(
            dataset_dir=cfg.dataset_dir,
            split=cfg.split,
            pred_labels_dir=pred_labels_dir,
            conf_th=float(cfg.conf_th),
            iou_th=float(cfg.iou_th),
            difficulties=diffs_attr,
            num_classes=cfg.num_classes,
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md: list[str] = []
    md.append("# Performance Analysis (KITTI-style)\n\n")
    md.append(f"Generated: `{now}`\n\n")
    md.append("## Protocol\n")
    md.append(
        "We evaluate object detection using PASCAL/VOC-style matching with a fixed IoU threshold, "
        "and report KITTI-like difficulty buckets. Since this dataset has no occlusion/truncation labels, "
        "difficulty is defined by minimum GT bounding-box height (in pixels) and, optionally, by per-box attributes "
        "(visibility/background) when enabled.\n\n"
    )
    md.append(f"- Dataset: `{cfg.dataset_dir}`\n")
    md.append(f"- Split: `{cfg.split}`\n")
    md.append(f"- Predictions: `{cfg.pred_dir}`\n")
    md.append(f"- Point confidence threshold: `{cfg.conf_th}`\n")
    md.append(f"- IoU threshold: `{cfg.iou_th}`\n")
    md.append("\n")
    md.append("### Dataset bbox-height statistics (GT)\n")
    if h.size:
        md.append(
            f"- Count: `{int(h.size)}`; min: `{hmin:.1f}px`; p25: `{float(np.quantile(h, 0.25)):.1f}px`; "
            f"p50: `{float(np.quantile(h, 0.50)):.1f}px`; p75: `{float(np.quantile(h, 0.75)):.1f}px`; max: `{hmax:.1f}px`\n"
        )
    else:
        md.append("- No GT boxes found to compute statistics.\n")
    md.append("\n")

    md.append("### Difficulty Buckets (auto if not specified)\n")
    md.append(f"- Easy: min bbox height `{easy_h}` px\n")
    md.append(f"- Moderate: min bbox height `{mod_h}` px\n")
    md.append(f"- Hard: min bbox height `{hard_h}` px\n")
    md.append("\n")

    md.append(_difficulty_summary_table(results_legacy))
    md.append(
        "**Notes (KITTI-like handling):**\n\n"
        "- GT boxes below the difficulty min-height are treated as *ignored* for that difficulty (not counted as FN).\n"
        "- Predictions below the difficulty min-height are ignored (not counted as FP).\n"
        "- Predictions that overlap an ignored GT of the same class at/above IoU are ignored (not counted as FP).\n"
        "- For AP/mAP, detections are ranked by confidence; a detection overlapping an ignored GT at/above IoU is ignored (neither TP nor FP).\n\n"
    )

    for r in results_legacy:
        md.append(_difficulty_section(r, class_names))

    if results_attr is not None:
        md.append("## Attribute-based Difficulty (new)\n\n")
        md.append(
            "These slices use percentile thresholds (0..1) over box-level attributes. "
            "They are intended to mimic KITTI-style difficulty using size/visibility/background filters.\n\n"
        )
        md.append(_difficulty_summary_table(results_attr))
        for r in results_attr:
            md.append(_difficulty_section(r, class_names))

    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.write_text("".join(md), encoding="utf-8")
    return cfg.out_md
