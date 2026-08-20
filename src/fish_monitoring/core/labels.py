"""Label filtering and area-based preprocessing.

Provides utilities to filter YOLO-format dataset splits by bounding-box
area, accounting for resolution rescaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AreaResize:
    original_width: int
    original_height: int
    new_width: int
    new_height: int


def adjusted_area_threshold(resize: AreaResize, original_area_thresh: float) -> float:
    scale_ratio = (resize.new_width * resize.new_height) / (resize.original_width * resize.original_height)
    return round(original_area_thresh * scale_ratio, 1)


def filter_labels_by_area(
    label_dir: Path,
    image_width: int,
    image_height: int,
    threshold_px2: float,
) -> tuple[dict[int, list[tuple[str, float]]], int]:
    """In-place filter YOLO labels by *pixel* area computed from normalized w/h.

    Returns (removed_boxes, total_removed) where removed_boxes is:
    {class_id: [(filename, area_px2), ...]}
    """
    removed_boxes: dict[int, list[tuple[str, float]]] = {}
    total_removed = 0

    for label_file in label_dir.glob("*.txt"):
        with label_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        kept_lines: list[str] = []
        removed_this_file: list[tuple[int, float]] = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                cls, _x, _y, w, h = map(float, parts)
            except ValueError:
                continue

            w_px = w * image_width
            h_px = h * image_height
            area = w_px * h_px

            if area >= threshold_px2:
                kept_lines.append(line)
            else:
                total_removed += 1
                removed_this_file.append((int(cls), round(area, 1)))

        label_file.write_text("".join(kept_lines), encoding="utf-8")

        for cls_id, area in removed_this_file:
            removed_boxes.setdefault(cls_id, []).append((label_file.name, area))

    return removed_boxes, total_removed


def filter_dataset_splits_by_area(
    dataset_base_dir: Path,
    splits: list[str],
    image_width: int,
    image_height: int,
    threshold_px2: float,
) -> dict[str, tuple[dict[int, list[tuple[str, float]]], int]]:
    """Runs `filter_labels_by_area` for each split's labels dir."""
    results: dict[str, tuple[dict[int, list[tuple[str, float]]], int]] = {}

    for split in splits:
        label_dir = dataset_base_dir / split / "labels"
        if not label_dir.exists():
            raise FileNotFoundError(f"No labels folder for split '{split}': {label_dir}")
        results[split] = filter_labels_by_area(label_dir, image_width, image_height, threshold_px2)

    return results
