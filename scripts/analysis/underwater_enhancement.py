#!/usr/bin/env python3
"""Underwater image enhancement preprocessing.

Applies simple image enhancement techniques designed for underwater
imagery and compares detection performance on raw vs enhanced images.

Enhancement methods implemented:
1. **CLAHE** — Contrast-Limited Adaptive Histogram Equalization
   Applied per-channel in LAB colour space.
2. **White-balance** — Grey-world assumption to correct colour cast.
3. **Sea-thru inspired** — simplified backscatter removal using the
   dark channel prior + colour restoration.

For each baseline, the script:
- Runs detection on raw test images → computes metrics.
- Enhances each test image → runs detection → computes metrics.
- Reports the delta.

Usage:
    python scripts/analysis/underwater_enhancement.py \
        --data   data/WIO-ReefFish/data.yaml \
        --results results/ \
        --split  test \
        --out    results/analysis/enhancement_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

from fish_monitoring.baselines.base_detector import (
    iter_split_images,
    read_yolo_labels,
    resolve_split_dirs,
)
from fish_monitoring.eval.diagnose import Gt, _load_image_size, _match_predictions
from fish_monitoring.core.inference import Pred

# ── Enhancement functions ────────────────────────────────────────────────

def _try_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError("opencv-python required for underwater enhancement")


def enhance_clahe(bgr: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
    """CLAHE on L-channel in LAB space."""
    cv2 = _try_cv2()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Grey-world white balance."""
    result = bgr.copy().astype(np.float32)
    avg_b = result[:, :, 0].mean()
    avg_g = result[:, :, 1].mean()
    avg_r = result[:, :, 2].mean()
    avg = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        result[:, :, 0] *= avg / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_sea_thru(bgr: np.ndarray, omega: float = 0.7, t_min: float = 0.1) -> np.ndarray:
    """Simplified Sea-thru: dark-channel prior dehazing + colour restoration.

    Steps:
    1. Estimate transmission via dark channel prior (inverted for underwater).
    2. Remove backscatter component.
    3. Apply white balance.
    """
    cv2 = _try_cv2()
    img = bgr.astype(np.float64)

    # Dark channel (patches of 15x15)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # For underwater: the "bright" channel (water absorbs red, so min is informative)
    min_channel = np.minimum(np.minimum(b, g), r)
    kernel = np.ones((15, 15), np.uint8)
    dark = cv2.erode(min_channel, kernel)

    # Atmospheric light estimate (top 0.1% brightest in dark channel)
    n_pixels = dark.size
    n_top = max(1, n_pixels // 1000)
    flat = dark.ravel()
    idx = np.argpartition(flat, -n_top)[-n_top:]
    atm_light = np.array([
        img[:, :, c].ravel()[idx].mean() for c in range(3)
    ])
    atm_light = np.maximum(atm_light, 1.0)

    # Transmission map
    normed = img / atm_light[np.newaxis, np.newaxis, :]
    min_normed = np.minimum(np.minimum(normed[:, :, 0], normed[:, :, 1]), normed[:, :, 2])
    t = 1.0 - omega * cv2.erode(min_normed, kernel)
    t = np.maximum(t, t_min)

    # Recover scene
    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - atm_light[c]) / t + atm_light[c]

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Final white balance
    result = enhance_white_balance(result)
    return result


ENHANCEMENT_METHODS = {
    "clahe": enhance_clahe,
    "white_balance": enhance_white_balance,
    "sea_thru": enhance_sea_thru,
}


# ── Evaluation helpers ───────────────────────────────────────────────────

def _eval_on_images(
    detector,
    image_paths: list[Path],
    labels_dir: Path,
    model_path: Path,
    imgsz: int,
    conf: float,
    iou_th: float,
    device: int,
    enhance_fn=None,
) -> dict[str, float]:
    """Run detection + eval on a list of images, optionally enhancing first."""
    cv2 = _try_cv2()

    tp_total = fp_total = fn_total = 0

    for img_path in image_paths:
        w, h = _load_image_size(img_path)

        if enhance_fn is not None:
            # Read, enhance, write to temp file
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            enhanced = enhance_fn(bgr)
            # Save to temp path
            tmp_path = img_path.parent / f"_enhanced_{img_path.name}"
            cv2.imwrite(str(tmp_path), enhanced)
            infer_path = tmp_path
        else:
            infer_path = img_path

        try:
            pred = detector.predict(
                infer_path,
                model_path=model_path,
                imgsz=imgsz,
                conf=conf,
                device=device,
            )
        finally:
            if enhance_fn is not None and infer_path.exists() and infer_path != img_path:
                infer_path.unlink(missing_ok=True)

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)
        gt = Gt(xyxy=gt_xyxy, cls=gt_cls)

        tp, fp, fn, _ = _match_predictions(gt, pred, iou_th=iou_th)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Underwater image enhancement analysis"
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=list(ENHANCEMENT_METHODS.keys()),
                        help="Enhancement methods to evaluate")
    parser.add_argument("--out", default="analysis/enhancement_analysis.csv")
    parser.add_argument("--baselines", nargs="*", default=None)
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    results_dir = Path(args.results).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from fish_monitoring.baselines.registry import list_baselines
    from fish_monitoring.baselines import get_baseline
    available = set(list_baselines())

    # Discover baselines
    baselines_to_eval: list[tuple[str, Path]] = []
    if args.baselines:
        for bl in args.baselines:
            for d in results_dir.iterdir():
                if d.is_dir() and (d.name.startswith(bl.replace("-", "_")) or d.name.startswith(bl)):
                    for wf in ["weights/best.pt", "best.pt"]:
                        if (d / wf).exists():
                            baselines_to_eval.append((bl, d / wf))
                            break
                    break
    else:
        for d in sorted(results_dir.iterdir()):
            if not d.is_dir():
                continue
            for bl in available:
                if d.name.startswith(bl.replace("-", "_")) or d.name.startswith(bl):
                    for wf in ["weights/best.pt", "best.pt"]:
                        if (d / wf).exists():
                            baselines_to_eval.append((bl, d / wf))
                            break
                    break

    _, labels_dir = resolve_split_dirs(data_yaml, args.split)
    image_paths = iter_split_images(data_yaml, args.split)

    rows = []

    print(f"\n{'='*80}")
    print(f"  Underwater Enhancement Analysis — {args.split} split")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"{'='*80}")

    for bl_name, model_path in baselines_to_eval:
        print(f"\n  ── {bl_name}  ({model_path})")
        detector = get_baseline(bl_name)

        # Raw (no enhancement)
        raw_metrics = _eval_on_images(
            detector, image_paths, labels_dir, model_path,
            args.imgsz, args.conf, args.iou, args.device,
            enhance_fn=None,
        )
        print(
            f"    raw:           P={raw_metrics['precision']:.3f}  "
            f"R={raw_metrics['recall']:.3f}  F1={raw_metrics['f1']:.3f}"
        )
        rows.append({
            "baseline": bl_name,
            "enhancement": "raw",
            "precision": f"{raw_metrics['precision']:.4f}",
            "recall": f"{raw_metrics['recall']:.4f}",
            "f1": f"{raw_metrics['f1']:.4f}",
            "delta_f1": "0.0000",
            "tp": raw_metrics["tp"],
            "fp": raw_metrics["fp"],
            "fn": raw_metrics["fn"],
        })

        for method_name in args.methods:
            enhance_fn = ENHANCEMENT_METHODS.get(method_name)
            if enhance_fn is None:
                print(f"    {method_name}: UNKNOWN — skipped")
                continue

            enh_metrics = _eval_on_images(
                detector, image_paths, labels_dir, model_path,
                args.imgsz, args.conf, args.iou, args.device,
                enhance_fn=enhance_fn,
            )
            delta = enh_metrics["f1"] - raw_metrics["f1"]
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
            print(
                f"    {method_name:>14s}:  P={enh_metrics['precision']:.3f}  "
                f"R={enh_metrics['recall']:.3f}  F1={enh_metrics['f1']:.3f}  "
                f"({arrow} {delta:+.3f})"
            )
            rows.append({
                "baseline": bl_name,
                "enhancement": method_name,
                "precision": f"{enh_metrics['precision']:.4f}",
                "recall": f"{enh_metrics['recall']:.4f}",
                "f1": f"{enh_metrics['f1']:.4f}",
                "delta_f1": f"{delta:+.4f}",
                "tp": enh_metrics["tp"],
                "fp": enh_metrics["fp"],
                "fn": enh_metrics["fn"],
            })

    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  ✓ Results saved to {out_path}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
