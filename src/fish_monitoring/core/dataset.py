"""Dataset loading and split statistics.

Utilities for counting images/labels per split and
per-class instance distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class SplitCounts:
    split: str
    images: int
    labels: int
    annotations: int


def count_split(base_path: Path, split: str) -> SplitCounts:
    images_path = base_path / split / "images"
    labels_path = base_path / split / "labels"

    image_files = [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    label_files = [p for p in labels_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    annotations = 0
    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            annotations += sum(1 for line in f if line.strip())

    return SplitCounts(split=split, images=len(image_files), labels=len(label_files), annotations=annotations)


def count_all_splits(base_path: Path, splits: Sequence[str]) -> list[SplitCounts]:
    return [count_split(base_path, split) for split in splits]


def print_split_counts(base_path: Path, splits: Sequence[str]) -> None:
    totals = SplitCounts(split="TOTAL", images=0, labels=0, annotations=0)

    for split in splits:
        c = count_split(base_path, split)
        print(f"\nSplit: {c.split}")
        print(f"  images:       {c.images}")
        print(f"  label files:  {c.labels}")
        print(f"  annotations:  {c.annotations}")

        totals = SplitCounts(
            split="TOTAL",
            images=totals.images + c.images,
            labels=totals.labels + c.labels,
            annotations=totals.annotations + c.annotations,
        )

    print("\nTOTAL across all splits")
    print(f"  images:       {totals.images}")
    print(f"  label files:  {totals.labels}")
    print(f"  annotations:  {totals.annotations}")


def class_instance_counts(
    base_path: Path,
    splits: Sequence[str],
    class_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Returns {split: {class_name: count}} counting instances from YOLO label files."""
    out: dict[str, dict[str, int]] = {}

    for split in splits:
        labels_dir = base_path / split / "labels"
        counts = {name: 0 for name in class_names}

        if not labels_dir.exists():
            out[split] = counts
            continue

        for label_file in labels_dir.iterdir():
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id_str = line.split()[0]
                    try:
                        cls_id = int(cls_id_str)
                    except ValueError:
                        continue
                    if 0 <= cls_id < len(class_names):
                        counts[class_names[cls_id]] += 1

        out[split] = counts

    return out


def print_class_instance_counts(
    base_path: Path,
    splits: Sequence[str],
    class_names: Sequence[str],
    limit: int | None = None,
) -> None:
    counts_by_split = class_instance_counts(base_path, splits, class_names)

    for split in splits:
        counts = counts_by_split[split]
        print(f"\n=== {split.upper()} SPLIT ===")

        items = list(counts.items())
        if limit is not None:
            items = items[:limit]

        total = 0
        for name, count in items:
            print(f"{name:<15}: {count}")
            total += count

        if limit is None:
            print(f"TOTAL instances: {total}")
        else:
            print(f"TOTAL instances (first {limit} classes): {total}")
