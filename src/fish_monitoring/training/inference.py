"""Final inference pipeline.

Runs a trained detector on a dataset split, optionally auto-selects
the best confidence threshold, and writes prediction labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fish_monitoring.eval.diagnose import pick_best_conf, sweep_conf
from fish_monitoring.core.inference import Pred, predict_image, predict_image_tiled


@dataclass(frozen=True)
class InferConfig:
    model_path: Path
    source: Path
    output_dir: Path
    imgsz: int = 640
    iou: float = 0.5
    device: int | str = 0

    # If conf is None, it will be selected automatically from calibration split.
    conf: float | None = None

    # Tiled inference
    tiled: bool = True
    tile_size: int = 1024
    tile_overlap: float = 0.25

    # Calibration dataset
    calib_dataset_dir: Path | None = None
    calib_split: str = "valid"
    calib_max_images: int | None = None
    calib_min_precision: float | None = None
    calib_min_recall: float | None = None
    calib_mode: str = "full"  # 'full' (fast) or 'tiled' (slow, matches inference)
    calib_save_csv: Path | None = None

    # Export PR/mAP artifacts (YOLO-like)
    export_metrics: bool = False
    metrics_dataset_dir: Path | None = None
    metrics_split: str = "test"
    pr_conf: float = 0.001  # low conf to keep detections for PR/mAP
    metrics_num_classes: int | None = None


def _iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png"}
        return sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    raise FileNotFoundError(f"Source not found: {source}")


def _try_read_dataset_nc(dataset_dir: Path) -> int | None:
    """Best-effort read of YOLO data.yaml 'nc' without requiring PyYAML."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return None

    for raw in data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nc:"):
            val = line.split(":", 1)[1].strip()
            try:
                return int(float(val))
            except ValueError:
                return None
    return None


def _guess_split_from_source(source: Path) -> str | None:
    parts = [p.lower() for p in source.resolve().parts]
    # Handle .../<split>/images or .../<split>/...
    for s in ("train", "valid", "val", "test"):
        if s in parts:
            return "valid" if s == "val" else s
    return None


def _save_yolo_txt(pred: Pred, image_path: Path, out_labels_dir: Path) -> Path:
    # Save predictions in YOLO txt format (cls x y w h conf) normalized.
    from PIL import Image

    out_labels_dir.mkdir(parents=True, exist_ok=True)
    w, h = Image.open(image_path).size

    lines: list[str] = []
    for (x1, y1, x2, y2), c, cls_id in zip(pred.xyxy, pred.conf, pred.cls):
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0

        # normalize
        x = cx / w
        y = cy / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {nw:.6f} {nh:.6f} {float(c):.6f}\n")

    out_path = out_labels_dir / f"{image_path.stem}.txt"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _save_annotated_image(pred: Pred, image_path: Path, out_images_dir: Path, class_names: list[str] | None) -> Path:
    import cv2

    out_images_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for (x1, y1, x2, y2), c, cls_id in zip(pred.xyxy, pred.conf, pred.cls):
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        name = str(int(cls_id))
        if class_names is not None and 0 <= int(cls_id) < len(class_names):
            name = class_names[int(cls_id)]
        cv2.putText(img, f"{name} {float(c):.2f}", (x1i, max(0, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = out_images_dir / image_path.name
    cv2.imwrite(str(out_path), img)
    return out_path


def auto_select_conf(*, model, cfg: InferConfig) -> float:
    if cfg.calib_dataset_dir is None:
        raise ValueError("calib_dataset_dir is required when conf is None")

    calib_tiled = str(cfg.calib_mode).strip().lower() == "tiled"

    # coarse-to-fine sweep
    coarse = [round(x, 2) for x in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]]
    points = sweep_conf(
        model=model,
        dataset_dir=cfg.calib_dataset_dir,
        split=cfg.calib_split,
        imgsz=cfg.imgsz,
        conf_values=coarse,
        iou=cfg.iou,
        device=cfg.device,
        tiled=calib_tiled,
        tile_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        max_images=cfg.calib_max_images,
    )
    best = pick_best_conf(points, min_precision=cfg.calib_min_precision, min_recall=cfg.calib_min_recall)

    # refine around best
    center = best.conf
    fine = sorted({round(x, 3) for x in [center - 0.05, center - 0.03, center - 0.02, center - 0.01, center, center + 0.01, center + 0.02, center + 0.03, center + 0.05] if 0.001 <= x <= 0.9})
    fine_points = sweep_conf(
        model=model,
        dataset_dir=cfg.calib_dataset_dir,
        split=cfg.calib_split,
        imgsz=cfg.imgsz,
        conf_values=fine,
        iou=cfg.iou,
        device=cfg.device,
        tiled=calib_tiled,
        tile_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        max_images=cfg.calib_max_images,
    )
    best2 = pick_best_conf(fine_points, min_precision=cfg.calib_min_precision, min_recall=cfg.calib_min_recall)

    if cfg.calib_save_csv is not None:
        import csv

        cfg.calib_save_csv.parent.mkdir(parents=True, exist_ok=True)
        with cfg.calib_save_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "conf", "precision", "recall", "f1", "tp", "fp", "fn"])
            for p in points:
                w.writerow(["coarse", p.conf, p.precision, p.recall, p.f1, p.tp, p.fp, p.fn])
            for p in fine_points:
                w.writerow(["fine", p.conf, p.precision, p.recall, p.f1, p.tp, p.fp, p.fn])

    return float(best2.conf)


def run_final_inference(cfg: InferConfig, *, save_txt: bool = True, save_images: bool = False) -> None:
    from ultralytics import YOLO

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_labels = cfg.output_dir / "labels"
    out_images = cfg.output_dir / "images"
    out_pr_labels = cfg.output_dir / "pr" / "labels"
    out_metrics = cfg.output_dir / "metrics"

    model = YOLO(str(cfg.model_path))

    # Helpful warning: dataset class-count mismatch often means you're using COCO weights.
    dataset_nc: int | None = None
    if cfg.export_metrics:
        guessed = _guess_split_from_source(cfg.source)
        if guessed is not None and str(cfg.metrics_split).strip().lower() != guessed:
            print(
                "WARNING: --metrics-split does not match the apparent split from --source. "
                f"source looks like '{guessed}', but metrics_split='{cfg.metrics_split}'. "
                "If they differ, exported metrics may be all zeros because GT/pred file stems won't match."
            )

        metrics_dataset = cfg.metrics_dataset_dir or cfg.calib_dataset_dir
        if metrics_dataset is not None:
            dataset_nc = _try_read_dataset_nc(Path(metrics_dataset))

        try:
            model_nc = len(model.names)  # type: ignore[arg-type]
        except Exception:
            model_nc = None

        if dataset_nc is not None and model_nc is not None and int(dataset_nc) != int(model_nc):
            print(
                "WARNING: Model/dataset class-count mismatch. "
                f"dataset nc={int(dataset_nc)} but model has {int(model_nc)} classes. "
                "If you run evaluation with COCO-pretrained weights (e.g. yolov8*.pt), TP will be ~0 because class IDs don't match. "
                "Use your trained weights for this dataset."
            )

    conf = cfg.conf
    if conf is None:
        conf = auto_select_conf(model=model, cfg=cfg)
        print(f"Selected conf={conf:.3f} (calib split={cfg.calib_split})")

    pr_conf = float(cfg.pr_conf)
    if pr_conf <= 0:
        pr_conf = 0.001
    if pr_conf > float(conf):
        pr_conf = float(conf)

    class_names: list[str] | None = None
    try:
        class_names = [model.names[i] for i in range(len(model.names))]
    except Exception:
        class_names = None

    image_paths = _iter_images(cfg.source)
    print(f"Images to process: {len(image_paths)}")

    for img_path in image_paths:
        if cfg.tiled:
            pred_all = predict_image_tiled(
                model,
                img_path,
                tile_size=cfg.tile_size,
                overlap=cfg.tile_overlap,
                imgsz=cfg.imgsz,
                conf=float(pr_conf),
                iou=cfg.iou,
                device=cfg.device,
            )
        else:
            pred_all = predict_image(model, img_path, imgsz=cfg.imgsz, conf=float(pr_conf), iou=cfg.iou, device=cfg.device)

        # Save PR-ready predictions (low conf, keeps confidences)
        if cfg.export_metrics:
            _save_yolo_txt(pred_all, img_path, out_pr_labels)

        # Final operating-point predictions
        if pred_all.conf.size:
            keep = pred_all.conf >= float(conf)
            pred = Pred(xyxy=pred_all.xyxy[keep], conf=pred_all.conf[keep], cls=pred_all.cls[keep])
        else:
            pred = pred_all

        if save_txt:
            _save_yolo_txt(pred, img_path, out_labels)
        if save_images:
            _save_annotated_image(pred, img_path, out_images, class_names)

    if cfg.export_metrics:
        from fish_monitoring.eval.metrics import export_yolo_like_artifacts

        metrics_dataset = cfg.metrics_dataset_dir or cfg.calib_dataset_dir
        if metrics_dataset is None:
            raise ValueError("metrics_dataset_dir (or calib_dataset_dir) is required when export_metrics=True")

        # Prefer dataset nc for confusion matrix axis (if not explicitly set).
        metrics_num_classes = cfg.metrics_num_classes
        if metrics_num_classes is None:
            metrics_num_classes = dataset_nc

        results = export_yolo_like_artifacts(
            dataset_dir=Path(metrics_dataset),
            split=str(cfg.metrics_split),
            pred_labels_dir=out_pr_labels,
            out_dir=out_metrics,
            point_conf=float(conf),
            iou_th=float(cfg.iou),
            num_classes=metrics_num_classes,
            class_names=class_names,
        )

        print("\nMetrics (by difficulty):")
        print("difficulty | min_h_px | precision | recall | mAP11 | mAP_cont")
        for r in results:
            print(
                f"{r.name:9s} | {int(r.min_height_px):8d} | {r.point.precision:9.4f} | {r.point.recall:6.4f} | {r.map_11:5.4f} | {r.map_continuous:8.4f}"
            )
        print(f"\nWrote YOLO-like artifacts under: {out_metrics}")
