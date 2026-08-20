#!/usr/bin/env python3
"""Generate per-sample TXT files containing BOTH visibility and background labels.

This produces one TXT file per image (sample), aligned with the YOLO label order.

Output structure (default):

  <dataset>/<split>/attributes/<image_stem>.txt

Each line corresponds to the same-index GT box in the YOLO label file:

  <box_index> <cls_id> <visibility> <background> <mean_v> <std_l> <edge_density> <blue_ratio>

Why TXT-per-sample?
- Easier to consume in downstream scripts without needing CSV joins.
- Keeps YOLO labels unchanged.

Example:
  python3 -m fish_monitoring.data_utils.label_attributes_txt \
    --dataset data/WIO-ReefFish --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fish_monitoring.data_utils.bbox_attributes import (
    compute_box_features,
    expand_xyxy,
    iter_dataset_images,
    load_bgr_image,
    quantize_background_state,
    quantize_visibility,
    read_yolo_boxes,
    yolo_xywhn_to_xyxy_px,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="label_attributes_txt")
    p.add_argument("--dataset", type=Path, required=True, help="YOLO dataset dir containing split/{images,labels}")
    p.add_argument("--split", default="valid", choices=["train", "valid", "test", "val"], help="Split to process")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for per-sample txt. Defaults to <dataset>/<split>/attributes",
    )
    p.add_argument("--expand", type=float, default=0.50, help="Expand bbox by this fraction before computing stats")

    # visibility
    p.add_argument("--dark-v-th", type=float, default=70.0)
    p.add_argument("--low-contrast-th", type=float, default=18.0)

    # background
    p.add_argument("--water-blue-th", type=float, default=0.36)
    p.add_argument("--edge-th", type=float, default=0.075)

    p.add_argument("--max-images", type=int, default=None, help="Optional limit for smoke tests")
    return p


def main() -> None:
    args = build_parser().parse_args()
    split = "valid" if args.split == "val" else args.split

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.dataset / split / "attributes"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_dataset_images(args.dataset, split)
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    n_written = 0
    n_boxes = 0

    for img_path in images:
        label_path = args.dataset / split / "labels" / f"{img_path.stem}.txt"
        boxes = read_yolo_boxes(label_path)
        if not boxes:
            # still write an empty attributes file to keep parity if needed
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            n_written += 1
            continue

        img = load_bgr_image(img_path)
        ih, iw = img.shape[:2]

        lines: list[str] = []
        for idx, b in enumerate(boxes):
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_px(b, im_w=iw, im_h=ih)
            x1e, y1e, x2e, y2e = expand_xyxy(x1, y1, x2, y2, expand=float(args.expand), w=iw, h=ih)
            crop = img[y1e:y2e, x1e:x2e]
            feat = compute_box_features(crop)

            vis = quantize_visibility(feat, dark_v_th=float(args.dark_v_th), low_contrast_th=float(args.low_contrast_th))
            bg = quantize_background_state(feat, water_blue_th=float(args.water_blue_th), edge_th=float(args.edge_th))

            # Keep extra stats: helps debugging / threshold tuning
            lines.append(
                f"{idx} {int(b.cls_id)} {vis} {bg} "
                f"{feat.mean_v:.2f} {feat.std_l:.2f} {feat.edge_density:.4f} {feat.blue_ratio:.4f}\n"
            )
            n_boxes += 1

        (out_dir / f"{img_path.stem}.txt").write_text("".join(lines), encoding="utf-8")
        n_written += 1

    print(f"Wrote {n_written} per-sample attribute files ({n_boxes} total boxes) -> {out_dir}")


if __name__ == "__main__":
    main()
