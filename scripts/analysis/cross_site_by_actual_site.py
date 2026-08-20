#!/usr/bin/env python3
"""Cross-site generalization analysis aggregated by actual reef site.

Uses split_per_site.txt to map video IDs → Site names, then aggregates
TP/FP/FN per site (not per video) to compute P/R/F1.

Usage:
    python scripts/analysis/cross_site_by_actual_site.py \
        --pred-cache results/eval_predictions \
        --data       data/WIO-ReefFish/data.yaml \
        --split      test \
        --site-map   data/WIO-ReefFish/metadata/split_per_site.txt \
        --baselines  rtdetr \
        --out        results/analysis/cross_site_by_site.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
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


# =====================================================================
#  Video → Site mapping
# =====================================================================

def load_site_map(path: Path) -> dict[str, dict]:
    """Parse split_per_site.txt → {video_id: {site, location, country}}."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Header row detection
            if parts[0].strip() == "" and "Video" in parts[1]:
                continue
            # Format: idx \t video_id \t site \t location \t country \t n_images
            if len(parts) < 6:
                continue
            try:
                int(parts[0])  # idx — skip header
            except ValueError:
                continue
            video_id = parts[1].strip()
            site = parts[2].strip()
            location = parts[3].strip()
            country = parts[4].strip()
            mapping[video_id] = {
                "site": site,
                "location": location,
                "country": country,
            }
    return mapping


def extract_video_id(stem: str) -> str:
    """Extract video ID prefix from an image filename stem.

    Examples:
        GX010034_frame_044_jpg.rf.xxx  →  GX010034
        Stephen_site1_T1_frame_003_... →  Stephen_site1_T1
    """
    m = re.match(r"^(.+?)_frame_", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


# =====================================================================
#  Helpers (from run_all_cached.py)
# =====================================================================

def load_cached_predictions(
    pred_dir: Path, img_stem: str, im_w: int, im_h: int,
) -> Pred:
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


# =====================================================================
#  Per-site analysis
# =====================================================================

def analyse_cross_site_by_actual_site(
    data_yaml: Path,
    split: str,
    pred_dir: Path,
    site_map: dict[str, dict],
    iou_th: float = 0.5,
) -> dict[str, dict]:
    """Per-site P/R/F1 using the split_per_site mapping."""
    _, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = iter_split_images(data_yaml, split)

    site_tp: dict[str, int] = defaultdict(int)
    site_fp: dict[str, int] = defaultdict(int)
    site_fn: dict[str, int] = defaultdict(int)
    site_n_img: dict[str, int] = defaultdict(int)
    site_meta: dict[str, dict] = {}
    unmapped = set()

    for img_path in image_paths:
        video_id = extract_video_id(img_path.stem)

        if video_id in site_map:
            info = site_map[video_id]
            site_name = info["site"]
            if site_name not in site_meta:
                site_meta[site_name] = {
                    "location": info["location"],
                    "country": info["country"],
                }
        else:
            unmapped.add(video_id)
            site_name = video_id  # fallback

        site_n_img[site_name] += 1

        w, h = _load_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)
        gt = Gt(xyxy=gt_xyxy, cls=gt_cls)

        pred = load_cached_predictions(pred_dir, img_path.stem, w, h)

        tp, fp, fn, _ = _match_predictions(gt, pred, iou_th=iou_th)
        site_tp[site_name] += tp
        site_fp[site_name] += fp
        site_fn[site_name] += fn

    if unmapped:
        print(f"  ⚠ Unmapped video IDs (used as-is): {unmapped}")

    results = {}
    for site_name in sorted(site_tp.keys() | site_fn.keys()):
        tp = site_tp[site_name]
        fp = site_fp[site_name]
        fn = site_fn[site_name]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        meta = site_meta.get(site_name, {"location": "?", "country": "?"})
        results[site_name] = {
            "site": site_name,
            "location": meta["location"],
            "country": meta["country"],
            "n_images": site_n_img[site_name],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-site analysis by actual site")
    parser.add_argument("--pred-cache", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--baselines", nargs="*", default=None,
                        help="Baselines to evaluate (default: all)")
    parser.add_argument("--out", default="results/analysis/cross_site_by_site.csv")
    args = parser.parse_args()

    pred_cache = Path(args.pred_cache).resolve()
    data_yaml = Path(args.data).resolve()
    site_map = load_site_map(Path(args.site_map).resolve())
    out_path = Path(args.out)

    print(f"Site map loaded: {len(site_map)} videos → "
          f"{len(set(v['site'] for v in site_map.values()))} sites")

    # Discover baselines
    all_baselines = []
    for d in sorted(pred_cache.iterdir()):
        if not d.is_dir():
            continue
        lbl_dir = d / args.split / "labels"
        if lbl_dir.is_dir() and any(lbl_dir.glob("*.txt")):
            all_baselines.append((d.name, lbl_dir))

    if args.baselines:
        selected = set(args.baselines)
        all_baselines = [(n, p) for n, p in all_baselines if n in selected]

    print(f"Baselines: {[n for n, _ in all_baselines]}")

    all_rows = []
    for bl_name, pred_dir in all_baselines:
        print(f"\n── {bl_name} ──")
        results = analyse_cross_site_by_actual_site(
            data_yaml, args.split, pred_dir, site_map, iou_th=args.iou,
        )

        print(f"  {'Site':<25s} {'Loc':<12s} {'Country':<8s} "
              f"{'n_img':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} "
              f"{'TP':>5s} {'FP':>5s} {'FN':>5s}")
        print("  " + "-" * 90)
        for site_name, r in results.items():
            print(f"  {r['site']:<25s} {r['location']:<12s} {r['country']:<8s} "
                  f"{r['n_images']:>5d} {r['precision']:>6.3f} {r['recall']:>6.3f} "
                  f"{r['f1']:>6.3f} {r['tp']:>5d} {r['fp']:>5d} {r['fn']:>5d}")
            all_rows.append({"baseline": bl_name, **r})

        # Summary stats
        f1_vals = [r["f1"] for r in results.values()]
        if f1_vals:
            mean_f1 = np.mean(f1_vals)
            std_f1 = np.std(f1_vals)
            print(f"\n  Summary: mean F1 = {mean_f1:.3f}, σ = {std_f1:.3f}, "
                  f"min = {min(f1_vals):.3f}, max = {max(f1_vals):.3f}, "
                  f"n_sites = {len(f1_vals)}")

    # Save CSV
    if all_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
