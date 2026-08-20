#!/usr/bin/env python3
"""Bounding-box attribute extraction utilities.

This module implements simple, dependency-light image statistics that can be
used to derive per-box attributes such as:
- visibility (is the fish in the dark / low-contrast area?)
- background state (water-like vs coral-like texture/color)

Design goals
- Works with the existing YOLO dataset format (image + label txt).
- Produces sidecar CSV files rather than modifying YOLO txt labels.
- Keeps features interpretable and fast (OpenCV + numpy).

The heuristics are intentionally simple so we can iterate quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class YoloBox:
    cls_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class BoxFeatures:
    # region stats
    mean_l: float
    std_l: float
    mean_s: float
    mean_v: float
    # edge density / texture proxy
    edge_density: float
    # color "blueness": water tends to skew blue/green
    blue_ratio: float
    green_ratio: float
    red_ratio: float


def read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    """Read YOLO label file (cls x y w h) normalized."""
    if not label_path.exists():
        return []

    out: list[YoloBox] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        out.append(YoloBox(cls_id=cls_id, x=x, y=y, w=w, h=h))
    return out


def yolo_xywhn_to_xyxy_px(b: YoloBox, *, im_w: int, im_h: int) -> tuple[int, int, int, int]:
    cx = b.x * im_w
    cy = b.y * im_h
    bw = b.w * im_w
    bh = b.h * im_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, *, w: int, h: int) -> tuple[int, int, int, int]:
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(0, min(w, x2))
    y2c = max(0, min(h, y2))
    if x2c <= x1c:
        x2c = min(w, x1c + 1)
    if y2c <= y1c:
        y2c = min(h, y1c + 1)
    return x1c, y1c, x2c, y2c


def expand_xyxy(x1: int, y1: int, x2: int, y2: int, *, expand: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Expand a bbox by a fraction of its size."""
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(bw * float(expand)))
    dy = int(round(bh * float(expand)))
    return clamp_xyxy(x1 - dx, y1 - dy, x2 + dx, y2 + dy, w=w, h=h)


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for bbox attribute scripts. Install opencv-python.") from e


def compute_box_features(bgr_crop: np.ndarray) -> BoxFeatures:
    cv2 = _try_import_cv2()

    if bgr_crop.size == 0:
        return BoxFeatures(
            mean_l=0.0,
            std_l=0.0,
            mean_s=0.0,
            mean_v=0.0,
            edge_density=0.0,
            blue_ratio=0.0,
            green_ratio=0.0,
            red_ratio=0.0,
        )

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    # OpenCV HSV channels: H[0..179], S[0..255], V[0..255]
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Luminance-like channel using Rec.601 (approx)
    b = bgr_crop[:, :, 0].astype(np.float32)
    g = bgr_crop[:, :, 1].astype(np.float32)
    r = bgr_crop[:, :, 2].astype(np.float32)
    l = 0.114 * b + 0.587 * g + 0.299 * r

    mean_l = float(l.mean())
    std_l = float(l.std())
    mean_s = float(s.mean())
    mean_v = float(v.mean())

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float((edges > 0).mean())

    sum_rgb = float(b.mean() + g.mean() + r.mean() + 1e-6)
    blue_ratio = float(b.mean() / sum_rgb)
    green_ratio = float(g.mean() / sum_rgb)
    red_ratio = float(r.mean() / sum_rgb)

    return BoxFeatures(
        mean_l=mean_l,
        std_l=std_l,
        mean_s=mean_s,
        mean_v=mean_v,
        edge_density=edge_density,
        blue_ratio=blue_ratio,
        green_ratio=green_ratio,
        red_ratio=red_ratio,
    )


def iter_dataset_images(dataset_dir: Path, split: str) -> list[Path]:
    images_dir = dataset_dir / split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_bgr_image(path: Path) -> np.ndarray:
    cv2 = _try_import_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def crop_region(img_bgr: np.ndarray, *, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return img_bgr[y1:y2, x1:x2]


def quantize_visibility(feat: BoxFeatures, *, dark_v_th: float, low_contrast_th: float) -> str:
    """Return visibility label: 'good' | 'medium' | 'poor'."""
    # Base darkness + contrast heuristic.
    # - mean_v low => dark
    # - std_l low => flat / low contrast
    dark = feat.mean_v < float(dark_v_th)
    low_contrast = feat.std_l < float(low_contrast_th)

    if dark and low_contrast:
        return "poor"
    if dark or low_contrast:
        return "medium"
    return "good"


def quantize_background_state(feat: BoxFeatures, *, water_blue_th: float, edge_th: float) -> str:
    """Return background label: 'water' | 'coral' | 'mixed'.

    Heuristic:
    - water: higher blue ratio + low edge density (low texture)
    - coral: higher edge density and not strongly blue
    - mixed: in between
    """
    looks_water = (feat.blue_ratio >= float(water_blue_th)) and (feat.edge_density < float(edge_th))
    looks_coral = (feat.edge_density >= float(edge_th)) and (feat.blue_ratio < float(water_blue_th))

    if looks_water:
        return "water"
    if looks_coral:
        return "coral"
    return "mixed"


def stable_box_id(image_stem: str, box_index: int) -> str:
    return f"{image_stem}#{box_index}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_float(x: float) -> str:
    return f"{x:.6f}"


def to_rows(
    *,
    image_path: Path,
    boxes: list[YoloBox],
    feats: list[BoxFeatures],
    visibility: list[str] | None = None,
    background: list[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, (b, f) in enumerate(zip(boxes, feats)):
        row: dict[str, object] = {
            "image": image_path.name,
            "stem": image_path.stem,
            "box_index": i,
            "box_id": stable_box_id(image_path.stem, i),
            "cls_id": int(b.cls_id),
            "x": float(b.x),
            "y": float(b.y),
            "w": float(b.w),
            "h": float(b.h),
            "mean_l": float(f.mean_l),
            "std_l": float(f.std_l),
            "mean_s": float(f.mean_s),
            "mean_v": float(f.mean_v),
            "edge_density": float(f.edge_density),
            "blue_ratio": float(f.blue_ratio),
            "green_ratio": float(f.green_ratio),
            "red_ratio": float(f.red_ratio),
        }
        if visibility is not None:
            row["visibility"] = str(visibility[i])
        if background is not None:
            row["background"] = str(background[i])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    import csv

    rows = list(rows)
    ensure_parent_dir(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
