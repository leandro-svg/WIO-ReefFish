"""Low-level inference helpers.

Provides the ``Pred`` dataclass and functions for running a detector
on single images or tiled crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Pred:
    # xyxy in absolute pixels
    xyxy: np.ndarray  # (N, 4)
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,) int


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> np.ndarray:
    """Pure numpy NMS. Returns indices to keep."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_th]

    return np.array(keep, dtype=np.int64)


def _pred_from_ultralytics(result) -> Pred:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    return Pred(xyxy=xyxy, conf=conf, cls=cls)


def predict_image(
    model,
    image_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
) -> Pred:
    res = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    return _pred_from_ultralytics(res)


def _tile_slices(length: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    assert 0 <= overlap < 1
    stride = max(1, int(round(tile * (1 - overlap))))
    if length <= tile:
        return [(0, length)]

    slices: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + tile, length)
        start = max(0, end - tile)
        slices.append((start, end))
        if end >= length:
            break
        start = start + stride
    # dedupe
    out: list[tuple[int, int]] = []
    for s in slices:
        if not out or out[-1] != s:
            out.append(s)
    return out


def predict_image_tiled(
    model,
    image_path: Path,
    *,
    tile_size: int,
    overlap: float,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    nms_iou: float = 0.6,
) -> Pred:
    """Tiled inference to improve small-object recall.

    - Splits image into overlapping tiles of size tile_size.
    - Runs YOLO on each tile.
    - Maps boxes back to full-image coords.
    - Runs a final NMS across all tiles.
    """
    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    # Also include a full-image prediction (helps large objects cut by tiles)
    full_res = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    full_pred = _pred_from_ultralytics(full_res)

    xs = _tile_slices(w, tile_size, overlap)
    ys = _tile_slices(h, tile_size, overlap)

    all_xyxy: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []

    if full_pred.xyxy.shape[0] > 0:
        all_xyxy.append(full_pred.xyxy)
        all_conf.append(full_pred.conf)
        all_cls.append(full_pred.cls)

    for y0, y1 in ys:
        for x0, x1 in xs:
            tile = im.crop((x0, y0, x1, y1))
            # Run predict on PIL image directly
            res = model.predict(
                source=tile,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            p = _pred_from_ultralytics(res)
            if p.xyxy.shape[0] == 0:
                continue

            # Map tile coords -> full image coords
            xyxy = p.xyxy.copy()
            xyxy[:, [0, 2]] += float(x0)
            xyxy[:, [1, 3]] += float(y0)

            all_xyxy.append(xyxy)
            all_conf.append(p.conf)
            all_cls.append(p.cls)

    if not all_xyxy:
        return Pred(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )

    xyxy = np.concatenate(all_xyxy, axis=0)
    confs = np.concatenate(all_conf, axis=0)
    clss = np.concatenate(all_cls, axis=0)

    # class-aware NMS (run NMS per class)
    keep_all: list[int] = []
    for c in np.unique(clss):
        idx = np.where(clss == c)[0]
        keep = _nms_xyxy(xyxy[idx], confs[idx], nms_iou)
        keep_all.extend(idx[keep].tolist())

    keep_all = np.array(sorted(set(keep_all)), dtype=np.int64)
    return Pred(xyxy=xyxy[keep_all], conf=confs[keep_all], cls=clss[keep_all])
