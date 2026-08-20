"""Diagnostic confidence-threshold sweeping.

Matches predictions to ground-truth and sweeps confidence thresholds
to find the best operating point (by F1 or mAP).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fish_monitoring.core.inference import Pred, predict_image, predict_image_tiled


@dataclass(frozen=True)
class Gt:
    # xyxy in absolute pixels
    xyxy: np.ndarray  # (M, 4)
    cls: np.ndarray  # (M,) int


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    im = Image.open(image_path)
    return im.size  # (w, h)


def _read_yolo_label_file(label_path: Path, *, w: int, h: int) -> Gt:
    if not label_path.exists():
        return Gt(xyxy=np.zeros((0, 4), dtype=np.float32), cls=np.zeros((0,), dtype=np.int64))

    xyxy: list[list[float]] = []
    cls: list[int] = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c = int(float(parts[0]))
            x, y, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        # YOLO normalized xywh -> xyxy pixels
        cx = x * w
        cy = y * h
        bw_px = bw * w
        bh_px = bh * h

        x1 = cx - bw_px / 2
        y1 = cy - bh_px / 2
        x2 = cx + bw_px / 2
        y2 = cy + bh_px / 2

        xyxy.append([x1, y1, x2, y2])
        cls.append(c)

    if not xyxy:
        return Gt(xyxy=np.zeros((0, 4), dtype=np.float32), cls=np.zeros((0,), dtype=np.int64))

    return Gt(xyxy=np.array(xyxy, dtype=np.float32), cls=np.array(cls, dtype=np.int64))


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes. a:(N,4), b:(M,4) -> (N,M)."""
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


def _match_predictions(gt: Gt, pred: Pred, iou_th: float) -> tuple[int, int, int, dict[int, tuple[int, int, int]]]:
    """Returns (tp, fp, fn, per_class) where per_class[c] = (tp, fp, fn)."""
    per_class: dict[int, tuple[int, int, int]] = {}

    if gt.xyxy.shape[0] == 0:
        # all predictions are FPs
        fp = int(pred.xyxy.shape[0])
        for c in np.unique(pred.cls):
            per_class[int(c)] = (0, int((pred.cls == c).sum()), 0)
        return 0, fp, 0, per_class

    if pred.xyxy.shape[0] == 0:
        fn = int(gt.xyxy.shape[0])
        for c in np.unique(gt.cls):
            per_class[int(c)] = (0, 0, int((gt.cls == c).sum()))
        return 0, 0, fn, per_class

    gt_used = np.zeros((gt.xyxy.shape[0],), dtype=bool)
    tp = 0
    fp = 0

    # Greedy match: sort preds by confidence
    order = np.argsort(-pred.conf)

    ious = _box_iou(pred.xyxy[order], gt.xyxy)  # (P, G)

    for pi, p_idx in enumerate(order):
        p_cls = int(pred.cls[p_idx])
        # only match GT of same class that are unused
        candidates = np.where((gt.cls == p_cls) & (~gt_used))[0]
        if candidates.size == 0:
            fp += 1
            tpc, fpc, fnc = per_class.get(p_cls, (0, 0, 0))
            per_class[p_cls] = (tpc, fpc + 1, fnc)
            continue

        best_j = candidates[np.argmax(ious[pi, candidates])]
        best_iou = float(ious[pi, best_j])
        if best_iou >= iou_th:
            gt_used[best_j] = True
            tp += 1
            tpc, fpc, fnc = per_class.get(p_cls, (0, 0, 0))
            per_class[p_cls] = (tpc + 1, fpc, fnc)
        else:
            fp += 1
            tpc, fpc, fnc = per_class.get(p_cls, (0, 0, 0))
            per_class[p_cls] = (tpc, fpc + 1, fnc)

    fn = int((~gt_used).sum())
    # distribute FN per class
    for c in np.unique(gt.cls):
        c = int(c)
        fnc = int(((gt.cls == c) & (~gt_used)).sum())
        tpc, fpc, _old_fnc = per_class.get(c, (0, 0, 0))
        per_class[c] = (tpc, fpc, _old_fnc + fnc)

    return tp, fp, fn, per_class


@dataclass(frozen=True)
class DiagnoseResult:
    images: int
    gt_boxes: int
    pred_boxes: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    per_class: dict[int, tuple[int, int, int]]
    size_bins: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SweepPoint:
    conf: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def diagnose(
    *,
    model,
    dataset_dir: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    tiled: bool,
    tile_size: int,
    tile_overlap: float,
    max_images: int | None,
) -> DiagnoseResult:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images is not None:
        image_paths = image_paths[: max_images]

    tp = fp = fn = 0
    gt_boxes = pred_boxes = 0
    per_class: dict[int, list[int]] = {}  # c -> [tp, fp, fn]

    # size bins based on GT area in pixels
    bins = {
        "small": {"tp": 0, "fn": 0, "gt": 0},
        "medium": {"tp": 0, "fn": 0, "gt": 0},
        "large": {"tp": 0, "fn": 0, "gt": 0},
    }

    for image_path in image_paths:
        w, h = _load_image_size(image_path)
        label_path = labels_dir / (image_path.stem + ".txt")
        gt_i = _read_yolo_label_file(label_path, w=w, h=h)

        if tiled:
            pred_i = predict_image_tiled(
                model,
                image_path,
                tile_size=tile_size,
                overlap=tile_overlap,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
            )
        else:
            pred_i = predict_image(model, image_path, imgsz=imgsz, conf=conf, iou=iou, device=device)

        tp_i, fp_i, fn_i, per_class_i = _match_predictions(gt_i, pred_i, iou_th=iou)

        tp += tp_i
        fp += fp_i
        fn += fn_i
        gt_boxes += int(gt_i.xyxy.shape[0])
        pred_boxes += int(pred_i.xyxy.shape[0])

        for c, (tpc, fpc, fnc) in per_class_i.items():
            arr = per_class.setdefault(int(c), [0, 0, 0])
            arr[0] += int(tpc)
            arr[1] += int(fpc)
            arr[2] += int(fnc)

        # size bins: attribute GT boxes as matched/unmatched
        if gt_i.xyxy.shape[0] > 0:
            # Determine which GT were matched by re-running match with bookkeeping
            # Simple approach: match per GT by best pred (class aware)
            matched = np.zeros((gt_i.xyxy.shape[0],), dtype=bool)
            if pred_i.xyxy.shape[0] > 0:
                order = np.argsort(-pred_i.conf)
                ious = _box_iou(pred_i.xyxy[order], gt_i.xyxy)
                used = np.zeros((gt_i.xyxy.shape[0],), dtype=bool)
                for pi, p_idx in enumerate(order):
                    p_cls = int(pred_i.cls[p_idx])
                    cand = np.where((gt_i.cls == p_cls) & (~used))[0]
                    if cand.size == 0:
                        continue
                    best = cand[np.argmax(ious[pi, cand])]
                    if float(ious[pi, best]) >= iou:
                        used[best] = True
                        matched[best] = True

            areas = (gt_i.xyxy[:, 2] - gt_i.xyxy[:, 0]).clip(min=0) * (gt_i.xyxy[:, 3] - gt_i.xyxy[:, 1]).clip(min=0)
            # thresholds in pixels^2 (tuned for 1920x1080-ish footage)
            for j, area in enumerate(areas):
                if area < 32 * 32:
                    b = "small"
                elif area < 96 * 96:
                    b = "medium"
                else:
                    b = "large"

                bins[b]["gt"] += 1
                if matched[j]:
                    bins[b]["tp"] += 1
                else:
                    bins[b]["fn"] += 1

    per_class_out: dict[int, tuple[int, int, int]] = {c: (v[0], v[1], v[2]) for c, v in per_class.items()}

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    size_bins = {
        k: {
            "recall": _safe_div(v["tp"], v["gt"]),
            "gt": float(v["gt"]),
            "tp": float(v["tp"]),
            "fn": float(v["fn"]),
        }
        for k, v in bins.items()
    }

    return DiagnoseResult(
        images=len(image_paths),
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        per_class=per_class_out,
        size_bins=size_bins,
    )


def sweep_conf(
    *,
    model,
    dataset_dir: Path,
    split: str,
    imgsz: int,
    conf_values: list[float],
    iou: float,
    device,
    tiled: bool,
    tile_size: int,
    tile_overlap: float,
    max_images: int | None,
) -> list[SweepPoint]:
    """Evaluate multiple confidence thresholds and return precision/recall/F1 points."""
    out: list[SweepPoint] = []
    for conf in conf_values:
        r = diagnose(
            model=model,
            dataset_dir=dataset_dir,
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_images=max_images,
        )
        f1 = (2 * r.precision * r.recall / (r.precision + r.recall)) if (r.precision + r.recall) else 0.0
        out.append(
            SweepPoint(
                conf=float(conf),
                precision=float(r.precision),
                recall=float(r.recall),
                f1=float(f1),
                tp=int(r.tp),
                fp=int(r.fp),
                fn=int(r.fn),
            )
        )
    return out


def pick_best_conf(
    points: list[SweepPoint],
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> SweepPoint:
    """Pick best threshold by F1 with optional constraints."""
    filtered = points
    if min_precision is not None:
        filtered = [p for p in filtered if p.precision >= min_precision]
    if min_recall is not None:
        filtered = [p for p in filtered if p.recall >= min_recall]
    if not filtered:
        # fall back to unconstrained
        filtered = points
    return max(filtered, key=lambda p: (p.f1, p.recall, p.precision))
