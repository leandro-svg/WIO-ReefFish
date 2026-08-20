"""YOLO-style metrics wrapper.

Thin convenience layer over ``kitti_eval`` that reads YOLO-format
label directories and returns mAP / per-class AP in a single call.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fish_monitoring.eval.kitti_eval import (
    BoxSet,
    DifficultyResult,
    evaluate_kitti_style,
    read_yolo_labels,
)


@dataclass(frozen=True)
class CurvePoint:
    conf: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    im = Image.open(image_path)
    return im.size


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    xx1 = np.maximum(ax1, bx1)
    yy1 = np.maximum(ay1, by1)
    xx2 = np.minimum(ax2, bx2)
    yy2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    a_area = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    b_area = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = a_area + b_area - inter + 1e-9
    return (inter / union).astype(np.float32)


def _heights_xyxy(xyxy: np.ndarray) -> np.ndarray:
    if xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (xyxy[:, 3] - xyxy[:, 1]).astype(np.float32)


def derive_difficulty_thresholds(*, dataset_dir: Path, split: str) -> tuple[int, int, int, dict[str, float]]:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    heights: list[float] = []
    for image_path in images:
        im_w, im_h = _load_image_size(image_path)
        gt_path = labels_dir / f"{image_path.stem}.txt"
        gt = read_yolo_labels(gt_path, im_w=im_w, im_h=im_h, has_conf=False)
        heights.extend(_heights_xyxy(gt.xyxy).astype(float).tolist())

    h = np.array(heights, dtype=np.float32)
    if h.size == 0:
        return 40, 25, 15, {"count": 0.0, "min": 0.0, "p20": 0.0, "p55": 0.0, "max": 0.0}

    # Target distribution: Easy 45% / Moderate 35% / Hard 20%
    #   Easy     = height >= P55  (top 45%)
    #   Moderate = height >= P20  (top 80%, i.e. 45% + 35%)
    #   Hard     = height >= 1    (all, bottom 20% are the "hard" additions)
    p20 = float(np.quantile(h, 0.20))
    p55 = float(np.quantile(h, 0.55))
    easy = max(1, int(round(p55)))
    moderate = max(1, int(round(min(p20, easy))))
    hard = 1  # include all boxes
    stats = {"count": float(h.size), "min": float(h.min(initial=0.0)), "p20": p20, "p55": p55, "max": float(h.max(initial=0.0))}
    return easy, moderate, hard, stats


def sweep_point_metrics(
    *,
    dataset_dir: Path,
    split: str,
    pred_labels_dir: Path,
    iou_th: float,
    min_height_px: int,
    conf_values: np.ndarray,
) -> list[CurvePoint]:
    images_dir = dataset_dir / split / "images"
    gt_labels_dir = dataset_dir / split / "labels"
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # (gt_xyxy, gt_cls, ign_xyxy, ign_cls, pr_xyxy, pr_cls, pr_conf)
    for image_path in images:
        im_w, im_h = _load_image_size(image_path)
        gt_path = gt_labels_dir / f"{image_path.stem}.txt"
        pr_path = pred_labels_dir / f"{image_path.stem}.txt"

        gt = read_yolo_labels(gt_path, im_w=im_w, im_h=im_h, has_conf=False)
        pr = read_yolo_labels(pr_path, im_w=im_w, im_h=im_h, has_conf=True)

        gt_h = _heights_xyxy(gt.xyxy)
        gt_keep = gt_h >= float(min_height_px)
        gt_ign = ~gt_keep
        gt_xyxy = gt.xyxy[gt_keep]
        gt_cls = gt.cls[gt_keep]
        ign_xyxy = gt.xyxy[gt_ign]
        ign_cls = gt.cls[gt_ign]

        if pr.conf is None:
            pr_conf = np.ones((pr.xyxy.shape[0],), dtype=np.float32)
        else:
            pr_conf = pr.conf

        if pr.xyxy.shape[0] == 0:
            pr_xyxy = np.zeros((0, 4), dtype=np.float32)
            pr_cls = np.zeros((0,), dtype=np.int64)
            pr_conf2 = np.zeros((0,), dtype=np.float32)
        else:
            pr_h = _heights_xyxy(pr.xyxy)
            pr_keep_size = pr_h >= float(min_height_px)
            pr_xyxy = pr.xyxy[pr_keep_size]
            pr_cls = pr.cls[pr_keep_size]
            pr_conf2 = pr_conf[pr_keep_size]

        per_image.append((gt_xyxy, gt_cls, ign_xyxy, ign_cls, pr_xyxy, pr_cls, pr_conf2))

    out: list[CurvePoint] = []

    for conf_th in conf_values.astype(float).tolist():
        tp = fp = fn = 0

        for gt_xyxy, gt_cls, ign_xyxy, ign_cls, pr_xyxy0, pr_cls0, pr_conf0 in per_image:
            pr_xyxy = pr_xyxy0
            pr_cls = pr_cls0
            pr_conf = pr_conf0

            keep_conf = pr_conf >= float(conf_th)
            pr_xyxy = pr_xyxy[keep_conf]
            pr_cls = pr_cls[keep_conf]
            pr_conf = pr_conf[keep_conf]

            # match using same logic as point evaluator (class-aware, greedy by conf)
            if gt_xyxy.shape[0] == 0:
                if pr_xyxy.shape[0] == 0:
                    continue
                if ign_xyxy.shape[0] == 0:
                    fp += int(pr_xyxy.shape[0])
                    continue

                ious_ign = _box_iou(pr_xyxy, ign_xyxy)
                for pi in range(pr_xyxy.shape[0]):
                    c = int(pr_cls[pi])
                    cand_ign = np.where(ign_cls == c)[0]
                    if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                        continue
                    fp += 1
                continue

            if pr_xyxy.shape[0] == 0:
                fn += int(gt_xyxy.shape[0])
                continue

            order = np.argsort(-pr_conf)
            ious = _box_iou(pr_xyxy[order], gt_xyxy)
            ious_ign = _box_iou(pr_xyxy[order], ign_xyxy) if ign_xyxy.shape[0] else None
            gt_used = np.zeros((gt_xyxy.shape[0],), dtype=bool)

            for pi, p_idx in enumerate(order):
                c = int(pr_cls[p_idx])
                cand = np.where((gt_cls == c) & (~gt_used))[0]
                if cand.size == 0:
                    if ious_ign is not None:
                        cand_ign = np.where(ign_cls == c)[0]
                        if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                            continue
                    fp += 1
                    continue

                best = cand[np.argmax(ious[pi, cand])]
                if float(ious[pi, best]) >= float(iou_th):
                    gt_used[best] = True
                    tp += 1
                else:
                    if ious_ign is not None:
                        cand_ign = np.where(ign_cls == c)[0]
                        if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                            continue
                    fp += 1

            fn += int((~gt_used).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out.append(CurvePoint(conf=float(conf_th), precision=precision, recall=recall, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn)))

    return out


def confusion_matrix_detection(
    *,
    dataset_dir: Path,
    split: str,
    pred_labels_dir: Path,
    conf_th: float,
    iou_th: float,
    min_height_px: int,
    num_classes: int,
) -> np.ndarray:
    images_dir = dataset_dir / split / "images"
    gt_labels_dir = dataset_dir / split / "labels"
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    bg = int(num_classes)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for image_path in images:
        im_w, im_h = _load_image_size(image_path)
        gt_path = gt_labels_dir / f"{image_path.stem}.txt"
        pr_path = pred_labels_dir / f"{image_path.stem}.txt"

        gt = read_yolo_labels(gt_path, im_w=im_w, im_h=im_h, has_conf=False)
        pr = read_yolo_labels(pr_path, im_w=im_w, im_h=im_h, has_conf=True)

        gt_h = _heights_xyxy(gt.xyxy)
        gt_keep = gt_h >= float(min_height_px)
        gt_ign = ~gt_keep
        gt_xyxy = gt.xyxy[gt_keep]
        gt_cls = gt.cls[gt_keep]
        ign_xyxy = gt.xyxy[gt_ign]
        ign_cls = gt.cls[gt_ign]

        if pr.conf is None:
            pr_conf = np.ones((pr.xyxy.shape[0],), dtype=np.float32)
        else:
            pr_conf = pr.conf

        pr_keep_conf = pr_conf >= float(conf_th)
        pr_xyxy0 = pr.xyxy[pr_keep_conf]
        pr_cls0 = pr.cls[pr_keep_conf]
        pr_conf0 = pr_conf[pr_keep_conf]

        pr_h = _heights_xyxy(pr_xyxy0)
        pr_keep_size = pr_h >= float(min_height_px)
        pr_xyxy = pr_xyxy0[pr_keep_size]
        pr_cls = pr_cls0[pr_keep_size]
        pr_conf1 = pr_conf0[pr_keep_size]

        if gt_xyxy.shape[0] == 0:
            if pr_xyxy.shape[0] == 0:
                continue
            if ign_xyxy.shape[0] == 0:
                for c in pr_cls:
                    if 0 <= int(c) < num_classes:
                        cm[bg, int(c)] += 1
                continue

            ious_ign = _box_iou(pr_xyxy, ign_xyxy)
            for pi in range(pr_xyxy.shape[0]):
                c = int(pr_cls[pi])
                if not (0 <= c < num_classes):
                    continue
                cand_ign = np.where(ign_cls == c)[0]
                if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                    continue
                cm[bg, c] += 1
            continue

        gt_used = np.zeros((gt_xyxy.shape[0],), dtype=bool)

        if pr_xyxy.shape[0] > 0:
            order = np.argsort(-pr_conf1)
            ious = _box_iou(pr_xyxy[order], gt_xyxy)
            ious_ign = _box_iou(pr_xyxy[order], ign_xyxy) if ign_xyxy.shape[0] else None

            for pi, p_idx in enumerate(order):
                pred_c = int(pr_cls[p_idx])
                if not (0 <= pred_c < num_classes):
                    continue

                # Find best GT (any class) not used
                cand = np.where(~gt_used)[0]
                if cand.size == 0:
                    # unmatched pred -> bg->pred, unless overlaps ignored GT of same class
                    if ious_ign is not None:
                        cand_ign = np.where(ign_cls == pred_c)[0]
                        if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                            continue
                    cm[bg, pred_c] += 1
                    continue

                best = cand[np.argmax(ious[pi, cand])]
                if float(ious[pi, best]) >= float(iou_th):
                    gt_used[best] = True
                    gt_c = int(gt_cls[best])
                    if 0 <= gt_c < num_classes:
                        cm[gt_c, pred_c] += 1
                else:
                    if ious_ign is not None:
                        cand_ign = np.where(ign_cls == pred_c)[0]
                        if cand_ign.size and float(ious_ign[pi, cand_ign].max(initial=0.0)) >= float(iou_th):
                            continue
                    cm[bg, pred_c] += 1

        # missed GT -> gt->bg
        for gi in np.where(~gt_used)[0]:
            gt_c = int(gt_cls[gi])
            if 0 <= gt_c < num_classes:
                cm[gt_c, bg] += 1

    return cm


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def export_yolo_like_artifacts(
    *,
    dataset_dir: Path,
    split: str,
    pred_labels_dir: Path,
    out_dir: Path,
    point_conf: float,
    iou_th: float,
    num_classes: int | None,
    class_names: list[str] | None = None,
) -> list[DifficultyResult]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check: ensure predictions exist for this evaluation split.
    images_dir = dataset_dir / split / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Split images dir not found: {images_dir}")

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise ValueError(f"No images found for split '{split}' under: {images_dir}")

    pred_dir = pred_labels_dir
    if pred_dir.is_dir() and (pred_dir / "labels").is_dir():
        pred_dir = pred_dir / "labels"

    present = 0
    for p in images:
        if (pred_dir / f"{p.stem}.txt").exists():
            present += 1

    coverage = present / max(1, len(images))
    if coverage < 0.5:
        raise ValueError(
            "Predictions do not match the requested metrics split. "
            f"Found {present}/{len(images)} prediction files for split='{split}' in '{pred_dir}'. "
            "This usually happens when you ran inference on a different split (e.g. source=test images) but set --metrics-split to another split (e.g. valid). "
            "Fix by running infer on the same split images or setting --metrics-split to match the inference source."
        )

    easy_h, moderate_h, hard_h, _stats = derive_difficulty_thresholds(dataset_dir=dataset_dir, split=split)
    difficulties = [("Overall", 1), ("Easy", easy_h), ("Moderate", moderate_h), ("Hard", hard_h)]

    results = evaluate_kitti_style(
        dataset_dir=dataset_dir,
        split=split,
        pred_labels_dir=pred_dir,
        conf_th=float(point_conf),
        iou_th=float(iou_th),
        difficulties=difficulties,
        num_classes=num_classes,
    )

    # results.csv (like YOLO, but per difficulty)
    rows: list[list[object]] = []
    for r in results:
        p = float(r.point.precision)
        rec = float(r.point.recall)
        f1 = _safe_div(2 * p * rec, p + rec)
        rows.append([r.name, r.min_height_px, r.point.tp, r.point.fp, r.point.fn, p, rec, f1, r.map_11, r.map_continuous])

    _write_csv(
        out_dir / "results.csv",
        ["difficulty", "min_h_px", "tp", "fp", "fn", "precision", "recall", "f1", "map11", "map_cont"],
        rows,
    )

    # curves + confusion matrix per difficulty
    conf_values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    plt = _try_import_matplotlib()

    for r in results:
        sub = out_dir / r.name.lower()
        sub.mkdir(parents=True, exist_ok=True)

        pts = sweep_point_metrics(
            dataset_dir=dataset_dir,
            split=split,
            pred_labels_dir=pred_dir,
            iou_th=float(iou_th),
            min_height_px=int(r.min_height_px),
            conf_values=conf_values,
        )

        _write_csv(
            sub / "curves.csv",
            ["conf", "precision", "recall", "f1", "tp", "fp", "fn"],
            [[p.conf, p.precision, p.recall, p.f1, p.tp, p.fp, p.fn] for p in pts],
        )

        if plt is not None:
            xs = np.array([p.conf for p in pts], dtype=np.float32)
            ps = np.array([p.precision for p in pts], dtype=np.float32)
            rs = np.array([p.recall for p in pts], dtype=np.float32)
            fs = np.array([p.f1 for p in pts], dtype=np.float32)

            plt.figure(figsize=(7, 5))
            plt.plot(xs, fs, label="F1")
            plt.xlabel("Confidence")
            plt.ylabel("F1")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(sub / "F1_curve.png", dpi=160, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(7, 5))
            plt.plot(xs, ps, label="Precision")
            plt.xlabel("Confidence")
            plt.ylabel("Precision")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(sub / "P_curve.png", dpi=160, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(7, 5))
            plt.plot(xs, rs, label="Recall")
            plt.xlabel("Confidence")
            plt.ylabel("Recall")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(sub / "R_curve.png", dpi=160, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(7, 5))
            plt.plot(rs, ps)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.savefig(sub / "PR_curve.png", dpi=160, bbox_inches="tight")
            plt.close()

        # confusion matrix at point conf
        if num_classes is None:
            if class_names is not None:
                nc = len(class_names)
            else:
                nc = int(max(list(r.pr_curves.keys()) + [0]) + 1)
        else:
            nc = int(num_classes)

        cm = confusion_matrix_detection(
            dataset_dir=dataset_dir,
            split=split,
            pred_labels_dir=pred_dir,
            conf_th=float(point_conf),
            iou_th=float(iou_th),
            min_height_px=int(r.min_height_px),
            num_classes=nc,
        )

        # normalized by row
        row_sum = cm.sum(axis=1, keepdims=True)
        cmn = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0.0)

        _write_csv(sub / "confusion_matrix_normalized.csv", ["row"] + [str(i) for i in range(nc)] + ["bg"], [[i] + cmn[i].astype(float).tolist() for i in range(nc + 1)])

        if plt is not None:
            labels = [str(i) for i in range(nc)] + ["bg"]
            if class_names is not None and len(class_names) == nc:
                labels = class_names + ["bg"]

            plt.figure(figsize=(10, 9))
            plt.imshow(cmn, vmin=0.0, vmax=1.0)
            plt.title(f"Confusion Matrix (normalized) - {r.name}")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(np.arange(nc + 1), labels, rotation=90, fontsize=7)
            plt.yticks(np.arange(nc + 1), labels, fontsize=7)
            plt.tight_layout()
            plt.savefig(sub / "confusion_matrix_normalized.png", dpi=200, bbox_inches="tight")
            plt.close()

    return results
