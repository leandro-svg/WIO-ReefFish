"""Spatial FP/FN analysis for object detection.

Goal
----
Help answer: "are my false negatives / false positives concentrated in specific
image regions (e.g. coral ground area)?"

Inputs
------
- YOLO dataset folder: <dataset>/<split>/{images,labels}
- Prediction labels folder: <pred_labels_dir>/*.txt in YOLO format
  (cls x y w h conf) normalized.

Outputs
-------
- CSV summary of FN/FP counts per grid cell
- Optional heatmap PNGs

This module is intentionally dependency-light. Heatmaps require matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fish_monitoring.eval.kitti_eval import _box_iou, _load_image_size, read_yolo_labels


@dataclass(frozen=True)
class SpatialReportConfig:
    dataset_dir: Path
    split: str
    pred_labels_dir: Path
    out_dir: Path
    conf_th: float = 0.25
    iou_th: float = 0.5
    min_height_px: int = 1
    grid_w: int = 3
    grid_h: int = 3
    # When True, also export per-image FN/FP points CSV
    export_points: bool = False


def _iter_images(dataset_dir: Path, split: str) -> list[Path]:
    images_dir = dataset_dir / split / "images"
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _read_pred_boxes(pred_path: Path, *, im_w: int, im_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read prediction YOLO txt (cls x y w h conf) -> (xyxy, cls, conf)."""
    pr = read_yolo_labels(pred_path, im_w=im_w, im_h=im_h, has_conf=True)
    conf = pr.conf if pr.conf is not None else np.ones((pr.xyxy.shape[0],), dtype=np.float32)
    return pr.xyxy, pr.cls, conf


def _center_xy(xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cx = ((xyxy[:, 0] + xyxy[:, 2]) * 0.5).astype(np.float32)
    cy = ((xyxy[:, 1] + xyxy[:, 3]) * 0.5).astype(np.float32)
    return cx, cy


def _bin_centers(cx: np.ndarray, cy: np.ndarray, *, im_w: int, im_h: int, grid_w: int, grid_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ix, iy) indices for each center into [0..grid_w-1], [0..grid_h-1]."""
    if cx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    x = np.clip(cx / max(float(im_w), 1.0), 0.0, 1.0 - 1e-9)
    y = np.clip(cy / max(float(im_h), 1.0), 0.0, 1.0 - 1e-9)
    ix = (x * float(grid_w)).astype(np.int64)
    iy = (y * float(grid_h)).astype(np.int64)
    ix = np.clip(ix, 0, grid_w - 1)
    iy = np.clip(iy, 0, grid_h - 1)
    return ix, iy


def _match_fp_fn(
    *,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    pr_xyxy: np.ndarray,
    pr_cls: np.ndarray,
    pr_conf: np.ndarray,
    conf_th: float,
    iou_th: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (fp_mask, fn_mask, fp_xyxy, fn_xyxy).

    Matching strategy: greedy by descending confidence, class-aware (similar to diagnose).
    """

    # Apply confidence threshold upfront
    keep = pr_conf >= float(conf_th)
    pr_xyxy = pr_xyxy[keep]
    pr_cls = pr_cls[keep]
    pr_conf = pr_conf[keep]

    if gt_xyxy.shape[0] == 0:
        fp_xyxy = pr_xyxy
        fn_xyxy = np.zeros((0, 4), dtype=np.float32)
        return np.ones((fp_xyxy.shape[0],), dtype=bool), np.zeros((0,), dtype=bool), fp_xyxy, fn_xyxy

    if pr_xyxy.shape[0] == 0:
        fp_xyxy = np.zeros((0, 4), dtype=np.float32)
        fn_xyxy = gt_xyxy
        return np.zeros((0,), dtype=bool), np.ones((fn_xyxy.shape[0],), dtype=bool), fp_xyxy, fn_xyxy

    order = np.argsort(-pr_conf)
    ious = _box_iou(pr_xyxy[order], gt_xyxy)

    gt_used = np.zeros((gt_xyxy.shape[0],), dtype=bool)
    fp_keep = np.zeros((pr_xyxy.shape[0],), dtype=bool)

    for rank, p_idx in enumerate(order):
        c = int(pr_cls[p_idx])
        cand = np.where((gt_cls == c) & (~gt_used))[0]
        if cand.size == 0:
            fp_keep[p_idx] = True
            continue
        best = cand[np.argmax(ious[rank, cand])]
        if float(ious[rank, best]) >= float(iou_th):
            gt_used[best] = True
        else:
            fp_keep[p_idx] = True

    fn_keep = ~gt_used
    fp_xyxy = pr_xyxy[fp_keep]
    fn_xyxy = gt_xyxy[fn_keep]
    return fp_keep, fn_keep, fp_xyxy, fn_xyxy


def _match_tp_fp_fn(
    *,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    pr_xyxy: np.ndarray,
    pr_cls: np.ndarray,
    pr_conf: np.ndarray,
    conf_th: float,
    iou_th: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (tp_xyxy, fp_xyxy, fn_xyxy, tp_gt_xyxy, pr_keep_mask, gt_used_mask).

    - tp_xyxy: prediction boxes that matched a GT
    - tp_gt_xyxy: the matched GT boxes for each TP (same length as tp_xyxy)
    - fp_xyxy: predictions not matched
    - fn_xyxy: GT not matched
    """

    # Apply confidence threshold upfront
    pr_keep_mask = pr_conf >= float(conf_th)
    pr_xyxy = pr_xyxy[pr_keep_mask]
    pr_cls = pr_cls[pr_keep_mask]
    pr_conf = pr_conf[pr_keep_mask]

    if gt_xyxy.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            pr_xyxy,
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            pr_keep_mask,
            np.zeros((0,), dtype=bool),
        )

    if pr_xyxy.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            gt_xyxy,
            np.zeros((0, 4), dtype=np.float32),
            pr_keep_mask,
            np.zeros((gt_xyxy.shape[0],), dtype=bool),
        )

    order = np.argsort(-pr_conf)
    ious = _box_iou(pr_xyxy[order], gt_xyxy)

    gt_used = np.zeros((gt_xyxy.shape[0],), dtype=bool)
    tp_list: list[np.ndarray] = []
    tp_gt_list: list[np.ndarray] = []
    fp_list: list[np.ndarray] = []

    for rank, p_sorted_idx in enumerate(order):
        c = int(pr_cls[p_sorted_idx])
        cand = np.where((gt_cls == c) & (~gt_used))[0]
        if cand.size == 0:
            fp_list.append(pr_xyxy[p_sorted_idx])
            continue
        best = cand[np.argmax(ious[rank, cand])]
        if float(ious[rank, best]) >= float(iou_th):
            gt_used[best] = True
            tp_list.append(pr_xyxy[p_sorted_idx])
            tp_gt_list.append(gt_xyxy[best])
        else:
            fp_list.append(pr_xyxy[p_sorted_idx])

    fn_xyxy = gt_xyxy[~gt_used]
    tp_xyxy = np.stack(tp_list, axis=0).astype(np.float32) if tp_list else np.zeros((0, 4), dtype=np.float32)
    tp_gt_xyxy = np.stack(tp_gt_list, axis=0).astype(np.float32) if tp_gt_list else np.zeros((0, 4), dtype=np.float32)
    fp_xyxy = np.stack(fp_list, axis=0).astype(np.float32) if fp_list else np.zeros((0, 4), dtype=np.float32)

    return tp_xyxy, fp_xyxy, fn_xyxy, tp_gt_xyxy, pr_keep_mask, gt_used


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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


def run_spatial_error_report(cfg: SpatialReportConfig) -> tuple[Path, Path | None, Path | None]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_images(cfg.dataset_dir, cfg.split)
    gt_labels_dir = cfg.dataset_dir / cfg.split / "labels"
    pred_labels_dir = cfg.pred_labels_dir

    # heatmaps
    fp_grid = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=np.int64)
    fn_grid = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=np.int64)
    tp_grid = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=np.int64)
    gt_grid = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=np.int64)
    # Tile-level TN: count of images where a given cell had no GT centers and no pred centers.
    tn_tile_grid = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=np.int64)

    point_rows: list[list[object]] = []

    for img_path in images:
        im_w, im_h = _load_image_size(img_path)

        gt_path = gt_labels_dir / f"{img_path.stem}.txt"
        pr_path = pred_labels_dir / f"{img_path.stem}.txt"

        gt = read_yolo_labels(gt_path, im_w=im_w, im_h=im_h, has_conf=False)
        pr_xyxy, pr_cls, pr_conf = _read_pred_boxes(pr_path, im_w=im_w, im_h=im_h)

        # Size gate on GT and preds (like KITTI difficulty)
        gt_h = (gt.xyxy[:, 3] - gt.xyxy[:, 1]).astype(np.float32) if gt.xyxy.size else np.zeros((0,), dtype=np.float32)
        gt_keep = gt_h >= float(cfg.min_height_px)
        gt_xyxy = gt.xyxy[gt_keep]
        gt_cls = gt.cls[gt_keep]

        pr_h = (pr_xyxy[:, 3] - pr_xyxy[:, 1]).astype(np.float32) if pr_xyxy.size else np.zeros((0,), dtype=np.float32)
        pr_keep = pr_h >= float(cfg.min_height_px)
        pr_xyxy2 = pr_xyxy[pr_keep]
        pr_cls2 = pr_cls[pr_keep]
        pr_conf2 = pr_conf[pr_keep]

        tp_xyxy, fp_xyxy, fn_xyxy, _tp_gt_xyxy, _pr_keep_mask, _gt_used = _match_tp_fp_fn(
            gt_xyxy=gt_xyxy,
            gt_cls=gt_cls,
            pr_xyxy=pr_xyxy2,
            pr_cls=pr_cls2,
            pr_conf=pr_conf2,
            conf_th=float(cfg.conf_th),
            iou_th=float(cfg.iou_th),
        )

        # Bin GT centers too (for normalization / "where are objects")
        gt_cx, gt_cy = _center_xy(gt_xyxy)
        gx, gy = _bin_centers(gt_cx, gt_cy, im_w=im_w, im_h=im_h, grid_w=int(cfg.grid_w), grid_h=int(cfg.grid_h))
        for ix, iy in zip(gx.tolist(), gy.tolist()):
            gt_grid[int(iy), int(ix)] += 1

        tp_cx, tp_cy = _center_xy(tp_xyxy)
        tx, ty = _bin_centers(tp_cx, tp_cy, im_w=im_w, im_h=im_h, grid_w=int(cfg.grid_w), grid_h=int(cfg.grid_h))
        for ix, iy in zip(tx.tolist(), ty.tolist()):
            tp_grid[int(iy), int(ix)] += 1
            if cfg.export_points:
                point_rows.append([img_path.name, "tp", int(ix), int(iy), float(ix + 0.5) / float(cfg.grid_w), float(iy + 0.5) / float(cfg.grid_h)])

        fp_cx, fp_cy = _center_xy(fp_xyxy)
        fx, fy = _bin_centers(fp_cx, fp_cy, im_w=im_w, im_h=im_h, grid_w=int(cfg.grid_w), grid_h=int(cfg.grid_h))
        for ix, iy in zip(fx.tolist(), fy.tolist()):
            fp_grid[int(iy), int(ix)] += 1
            if cfg.export_points:
                point_rows.append([img_path.name, "fp", int(ix), int(iy), float(ix + 0.5) / float(cfg.grid_w), float(iy + 0.5) / float(cfg.grid_h)])

        fn_cx, fn_cy = _center_xy(fn_xyxy)
        nx, ny = _bin_centers(fn_cx, fn_cy, im_w=im_w, im_h=im_h, grid_w=int(cfg.grid_w), grid_h=int(cfg.grid_h))
        for ix, iy in zip(nx.tolist(), ny.tolist()):
            fn_grid[int(iy), int(ix)] += 1
            if cfg.export_points:
                point_rows.append([img_path.name, "fn", int(ix), int(iy), float(ix + 0.5) / float(cfg.grid_w), float(iy + 0.5) / float(cfg.grid_h)])

        # Tile-level TN per image: mark cells containing any GT center or pred center;
        # remaining cells count as TN for that image.
        cell_has_gt = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=bool)
        cell_has_pr = np.zeros((int(cfg.grid_h), int(cfg.grid_w)), dtype=bool)
        for ix, iy in zip(gx.tolist(), gy.tolist()):
            cell_has_gt[int(iy), int(ix)] = True

        pr_cent_xyxy = pr_xyxy2[pr_conf2 >= float(cfg.conf_th)]
        pr_cx, pr_cy = _center_xy(pr_cent_xyxy)
        px, py = _bin_centers(pr_cx, pr_cy, im_w=im_w, im_h=im_h, grid_w=int(cfg.grid_w), grid_h=int(cfg.grid_h))
        for ix, iy in zip(px.tolist(), py.tolist()):
            cell_has_pr[int(iy), int(ix)] = True

        tn_tile_grid += (~cell_has_gt & ~cell_has_pr).astype(np.int64)

    # Write CSV summary
    summary_rows: list[list[object]] = []
    for iy in range(int(cfg.grid_h)):
        for ix in range(int(cfg.grid_w)):
            gt_n = int(gt_grid[iy, ix])
            tp_n = int(tp_grid[iy, ix])
            fp_n = int(fp_grid[iy, ix])
            fn_n = int(fn_grid[iy, ix])
            tn_tile_n = int(tn_tile_grid[iy, ix])
            fp_per_gt = float(fp_n / gt_n) if gt_n else float(fp_n)
            fn_per_gt = float(fn_n / gt_n) if gt_n else float(fn_n)
            tp_recall = float(tp_n / gt_n) if gt_n else 0.0
            summary_rows.append([ix, iy, gt_n, tp_n, fp_n, fn_n, tn_tile_n, tp_recall, fp_per_gt, fn_per_gt])

    summary_csv = cfg.out_dir / f"spatial_errors_{cfg.split}.csv"
    _write_csv(
        summary_csv,
        [
            "cell_x",
            "cell_y",
            "gt",
            "tp",
            "fp",
            "fn",
            "tn_tiles",
            "tp_recall",
            "fp_per_gt",
            "fn_per_gt",
        ],
        summary_rows,
    )

    points_csv: Path | None = None
    if cfg.export_points:
        points_csv = cfg.out_dir / f"spatial_error_points_{cfg.split}.csv"
        _write_csv(points_csv, ["image", "type", "cell_x", "cell_y", "u", "v"], point_rows)

    # Heatmaps
    plt = _try_import_matplotlib()
    fp_png: Path | None = None
    fn_png: Path | None = None
    tp_png: Path | None = None
    tn_tiles_png: Path | None = None
    if plt is not None:
        def _save_heatmap(arr: np.ndarray, title: str, out_path: Path) -> None:
            plt.figure(figsize=(6, 5))
            plt.imshow(arr, cmap="magma")
            plt.title(title)
            plt.xlabel("grid x")
            plt.ylabel("grid y")
            plt.colorbar()
            plt.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200)
            plt.close()

        fp_png = cfg.out_dir / f"fp_heatmap_{cfg.split}.png"
        fn_png = cfg.out_dir / f"fn_heatmap_{cfg.split}.png"
        tp_png = cfg.out_dir / f"tp_heatmap_{cfg.split}.png"
        tn_tiles_png = cfg.out_dir / f"tn_tiles_heatmap_{cfg.split}.png"
        _save_heatmap(fp_grid.astype(np.float32), f"FP heatmap ({cfg.split})", fp_png)
        _save_heatmap(fn_grid.astype(np.float32), f"FN heatmap ({cfg.split})", fn_png)
        _save_heatmap(tp_grid.astype(np.float32), f"TP heatmap ({cfg.split})", tp_png)
        _save_heatmap(tn_tile_grid.astype(np.float32), f"TN tiles heatmap ({cfg.split})", tn_tiles_png)

    return summary_csv, fp_png, fn_png
