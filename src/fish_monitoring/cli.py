"""Command-line interface for fish_monitoring.

Provides sub-commands for training, validation, inference, evaluation,
dataset inspection, diagnostics, and video rendering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fish_monitoring.constants import CLASS_NAMES, DEFAULT_SPLITS
from fish_monitoring.core.dataset import print_class_instance_counts, print_split_counts
from fish_monitoring.eval.diagnose import diagnose
from fish_monitoring.training.inference import InferConfig, run_final_inference
from fish_monitoring.core.labels import AreaResize, adjusted_area_threshold, filter_dataset_splits_by_area
from fish_monitoring.training.video import VideoArgs, run_video
from fish_monitoring.underwater.ultralytics_integration import UnderwaterTrainConfig
from fish_monitoring.training.yolo_ops import TrainArgs, ValArgs, checks, print_cuda_info, train, train_with_fallback, validate


def _repo_root() -> Path:
    # <repo>/src/fish_monitoring/cli.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _default_dataset_dir() -> Path:
    """Fallback dataset location when ``--data`` is not given."""
    return _repo_root() / "data" / "WIO-ReefFish"


def _resolve_data_yaml(p: Path) -> Path:
    """Resolve a YOLO dataset `data.yaml` path.

    Accepts either a direct path to a yaml file or a dataset directory.
    Also tries common repo-local locations to reduce CWD confusion.
    """

    p = Path(p)
    if p.is_dir():
        p = p / "data.yaml"

    if p.exists():
        return p

    # If the user passed something like "../dataset_x/data.yaml" but the repo actually
    # stores datasets under "data/<name>/", infer the dataset folder name.
    dataset_name: str | None = None
    if p.name.lower() in {"data.yaml", "data.yml"}:
        dataset_name = p.parent.name
    else:
        # If a directory path was provided but doesn't exist from current CWD,
        # use its last path component as dataset name.
        dataset_name = p.name

    root = _repo_root()
    candidates = [
        root / p,
        root / "datasets" / p,
        root / "src" / p,
        root / "src" / "datasets" / p,
    ]

    if dataset_name:
        # Direct dataset folder lookups.
        candidates.extend(
            [
                root / dataset_name / "data.yaml",
                root / "datasets" / dataset_name / "data.yaml",
                root / "src" / "datasets" / dataset_name / "data.yaml",
                root / "src" / "datasets" / dataset_name / "data.yml",
            ]
        )

    for c in candidates:
        if c.is_dir():
            c = c / "data.yaml"
        if c.exists():
            return c

    tried = "\n".join([str(p)] + [str(x) for x in candidates])
    raise FileNotFoundError(f"Dataset yaml not found. Tried:\n{tried}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fish-monitoring")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("stats", help="Count images/labels/annotations per split")
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))

    sp = sub.add_parser("class-counts", help="Count instances per class from labels")
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    sp.add_argument("--limit", type=int, default=None, help="Only print first N classes")

    sp = sub.add_parser("checks", help="Run ultralytics environment checks")

    sp = sub.add_parser("cuda-info", help="Print CUDA GPU info (torch)")

    sp = sub.add_parser("train", help="Train YOLO")
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--weights", default="yolov8m.pt")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=32)
    sp.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (0..1). Useful for smoke tests.",
    )
    sp.add_argument(
        "--device",
        default="0",
        help="Device spec passed to ultralytics (e.g. 0, 'cuda', '0,1')",
    )
    sp.add_argument("--patience", type=int, default=20)
    sp.add_argument("--project", default="results")
    sp.add_argument("--name", default="run")
    sp.add_argument("--no-val", action="store_true")
    sp.add_argument("--underwater-udino", action="store_true", help="Enable UDINO-style refinement preprocessor (experimental)")
    sp.add_argument(
        "--udino-radii",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.45],
        help="High-pass radii as fractions of max frequency (only if --underwater-udino)",
    )
    sp.add_argument(
        "--udino-gate-reduction",
        type=int,
        default=16,
        help="Channel-gate reduction ratio (only if --underwater-udino)",
    )
    sp.add_argument("--underwater-fwnwd", action="store_true", help="Use FWNWD bbox loss (experimental; patches Ultralytics loss)")
    sp.add_argument("--fwnwd-gamma", type=float, default=2.0)
    sp.add_argument("--fwnwd-wiou-k", type=float, default=1.0)
    sp.add_argument("--fwnwd-nwd-lambda", type=float, default=1.0)
    sp.add_argument("--fwnwd-normalizer", type=float, default=200.0)
    sp.add_argument(
        "--traceback-path",
        type=Path,
        default=_repo_root() / "train_error_traceback.txt",
    )
    sp.add_argument("--visibility-head", action="store_true", help="Enable auxiliary visibility prediction head (requires attribute labels)")
    sp.add_argument("--vis-loss-weight", type=float, default=0.5, help="Weight for the visibility auxiliary loss")

    sp = sub.add_parser("train-fallback", help="Train YOLO with fallback on failure")
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--weights", default="yolov8m.pt")
    sp.add_argument("--epochs", type=int, default=150)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=32)
    sp.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (0..1). Useful for smoke tests.",
    )
    sp.add_argument("--patience", type=int, default=20)
    sp.add_argument("--project", default="results")
    sp.add_argument("--primary-name", default="run_primary")
    sp.add_argument("--fallback-name", default="run_fallback")
    sp.add_argument("--primary-device", default="0,1")
    sp.add_argument("--fallback-device", default="0")
    sp.add_argument("--underwater-udino", action="store_true", help="Enable UDINO-style refinement preprocessor (experimental)")
    sp.add_argument(
        "--udino-radii",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.45],
        help="High-pass radii as fractions of max frequency (only if --underwater-udino)",
    )
    sp.add_argument(
        "--udino-gate-reduction",
        type=int,
        default=16,
        help="Channel-gate reduction ratio (only if --underwater-udino)",
    )
    sp.add_argument("--underwater-fwnwd", action="store_true", help="Use FWNWD bbox loss (experimental; patches Ultralytics loss)")
    sp.add_argument("--fwnwd-gamma", type=float, default=2.0)
    sp.add_argument("--fwnwd-wiou-k", type=float, default=1.0)
    sp.add_argument("--fwnwd-nwd-lambda", type=float, default=1.0)
    sp.add_argument("--fwnwd-normalizer", type=float, default=200.0)
    sp.add_argument(
        "--traceback-path",
        type=Path,
        default=_repo_root() / "train_error_traceback.txt",
    )

    sp = sub.add_parser("eval", help="Evaluate a model on val/test")
    sp.add_argument("--model", type=Path, required=True)
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--split", default="val", choices=["train", "val", "test"])
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument(
        "--device",
        default="0",
        help="Device spec passed to ultralytics val (e.g. 0, 1, 'cuda', '0,1')",
    )
    sp.add_argument("--project", default="results")
    sp.add_argument("--name", default="eval")
    sp.add_argument(
        "--kitti-report",
        type=Path,
        default=None,
        help="If set, also export prediction labels and write our KITTI-style difficulty report markdown to this path.",
    )
    sp.add_argument(
        "--pred-out",
        type=Path,
        default=None,
        help="Where to export predictions (infer outputs). Defaults to <project>/<name>_preds.",
    )
    sp.add_argument("--conf", type=float, default=0.25, help="Confidence threshold used for exported predictions + point metrics")
    sp.add_argument("--iou", type=float, default=0.5, help="IoU threshold for our report")
    sp.add_argument("--easy-min-h", type=int, default=None)
    sp.add_argument("--moderate-min-h", type=int, default=None)
    sp.add_argument("--hard-min-h", type=int, default=None)
    sp.add_argument("--num-classes", type=int, default=None)
    sp.add_argument("--size-pct", type=float, default=None)
    sp.add_argument("--visibility-pct", type=float, default=None)
    sp.add_argument("--background-pct", type=float, default=None)

    sp = sub.add_parser("filter-labels", help="Filter YOLO labels by box area threshold")
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    sp.add_argument("--original-width", type=int, default=1920)
    sp.add_argument("--original-height", type=int, default=1080)
    sp.add_argument("--new-width", type=int, default=640)
    sp.add_argument("--new-height", type=int, default=640)
    sp.add_argument("--original-area-thresh", type=float, default=500)

    sp = sub.add_parser("video", help="Run tracking on a video and render MP4")
    sp.add_argument("--weights", type=Path, required=True)
    sp.add_argument("--input", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--conf", type=float, default=0.168)
    sp.add_argument("--tracker", default="bytetrack.yaml")
    sp.add_argument("--seconds", type=float, default=None)
    sp.add_argument("--device", default="0")
    sp.add_argument("--crf", type=int, default=18)
    sp.add_argument("--preset", default="fast")

    sp = sub.add_parser(
        "diagnose",
        help="Quantify misses (predicted background) and compare tiled inference",
    )
    sp.add_argument("--model", type=Path, required=True, help="Path to best.pt/last.pt")
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    sp.add_argument("--device", default="0")
    sp.add_argument("--max-images", type=int, default=None)
    sp.add_argument("--tiled", action="store_true", help="Use tiled inference")
    sp.add_argument("--tile-size", type=int, default=1024)
    sp.add_argument("--tile-overlap", type=float, default=0.25)

    sp = sub.add_parser(
        "infer",
        help="Final inference: tiled+full merge + optional auto conf calibration",
    )
    sp.add_argument("--model", type=Path, required=True)
    sp.add_argument("--source", type=Path, required=True, help="Image file or folder")
    sp.add_argument("--out", type=Path, required=True, help="Output folder")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--device", default="0")
    sp.add_argument(
        "--conf",
        default="auto",
        help="Confidence threshold (float) or 'auto' to select from validation split",
    )
    sp.add_argument("--tiled", action="store_true", help="Use tiled inference (recommended)")
    sp.add_argument("--tile-size", type=int, default=1024)
    sp.add_argument("--tile-overlap", type=float, default=0.25)
    sp.add_argument("--calib-dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--calib-split", default="valid", choices=["train", "valid", "test"])
    sp.add_argument("--calib-max-images", type=int, default=None)
    sp.add_argument(
        "--calib-mode",
        default="full",
        choices=["full", "tiled"],
        help="How to run auto-calibration: 'full' (fast) or 'tiled' (slow; matches --tiled)",
    )
    sp.add_argument(
        "--calib-save-csv",
        type=Path,
        default=None,
        help="Save the auto-calibration sweep (coarse+fine) as CSV",
    )
    sp.add_argument("--min-precision", type=float, default=None)
    sp.add_argument("--min-recall", type=float, default=None)
    sp.add_argument("--save-images", action="store_true", help="Save annotated images (requires opencv)")
    sp.add_argument(
        "--export-metrics",
        action="store_true",
        help="Also export YOLO-like curves (F1/P/R/PR), normalized confusion matrix, and results.csv under out/metrics (uses PR-ready low-conf predictions)",
    )
    sp.add_argument(
        "--metrics-dataset",
        type=Path,
        default=None,
        help="Dataset dir for metrics export (defaults to --calib-dataset)",
    )
    sp.add_argument(
        "--metrics-split",
        default="test",
        choices=["train", "valid", "test"],
        help="Split for metrics export (default: test)",
    )
    sp.add_argument(
        "--pr-conf",
        type=float,
        default=0.001,
        help="Low confidence used to export PR-ready predictions (default: 0.001). Final labels are filtered to the chosen --conf.",
    )
    sp.add_argument(
        "--metrics-num-classes",
        type=int,
        default=None,
        help="Override number of classes for confusion matrix ticks (default: inferred from model names)",
    )

    sp = sub.add_parser(
        "report",
        help="Write KITTI-style performance report (TP/FP/FN, precision/recall, mAP) from saved predictions",
    )
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--split", default="test", choices=["train", "valid", "test"])
    sp.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Predictions folder (either infer output folder or its labels/ subfolder)",
    )
    sp.add_argument("--out", type=Path, default=Path("performance_analysis.md"))
    sp.add_argument("--conf", type=float, default=0.52, help="Confidence threshold for point metrics")
    sp.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    sp.add_argument("--easy-min-h", type=int, default=None, help="Override Easy min height (px); default auto from GT stats")
    sp.add_argument("--moderate-min-h", type=int, default=None, help="Override Moderate min height (px); default auto from GT stats")
    sp.add_argument("--hard-min-h", type=int, default=None, help="Override Hard min height (px); default auto from GT stats")
    sp.add_argument("--num-classes", type=int, default=None)
    sp.add_argument(
        "--size-pct",
        type=float,
        default=None,
        help="Enable attribute difficulty: minimum size percentile (0..1) for 'Size' and 'Combined' slices",
    )
    sp.add_argument(
        "--visibility-pct",
        type=float,
        default=None,
        help="Enable attribute difficulty: minimum visibility score percentile (0..1) for 'Visibility' and 'Combined' slices",
    )
    sp.add_argument(
        "--background-pct",
        type=float,
        default=None,
        help="Enable attribute difficulty: minimum background score percentile (0..1) for 'Background' and 'Combined' slices",
    )

    sp = sub.add_parser(
        "spatial-errors",
        help="Analyze where FPs/FNs happen spatially (grid heatmap + CSV)",
    )
    sp.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root")
    sp.add_argument("--split", default="test", choices=["train", "valid", "test"])
    sp.add_argument(
        "--pred-labels",
        type=Path,
        required=True,
        help="Folder with prediction .txt files in YOLO format (cls x y w h conf)",
    )
    sp.add_argument("--out", type=Path, required=True, help="Output directory")
    sp.add_argument("--grid-w", type=int, default=3)
    sp.add_argument("--grid-h", type=int, default=3)
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--min-height", type=int, default=1)
    sp.add_argument("--export-points", action="store_true")

    sp = sub.add_parser(
        "underwater",
        help="Experimental underwater modules (FWNWD loss, UDINO refinement)",
    )
    sp.add_argument("args", nargs=argparse.REMAINDER, help="Forwarded to fish_monitoring.underwater.demo")

    # -------------------------------------------------------------------
    # Baseline detectors
    # -------------------------------------------------------------------
    from fish_monitoring.baselines.registry import list_baselines as _list_baselines
    _baseline_names = ", ".join(_list_baselines())

    sp = sub.add_parser(
        "train-baseline",
        help=f"Train a baseline detector ({_baseline_names})",
    )
    sp.add_argument("--baseline", required=True, help=f"Baseline name: {_baseline_names}")
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--weights", default="", help="Pretrained weights (model-specific)")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default="0")
    sp.add_argument("--patience", type=int, default=20)
    sp.add_argument("--project", default="results")
    sp.add_argument("--name", default=None, help="Run name (defaults to baseline name)")
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))

    sp = sub.add_parser(
        "eval-baseline",
        help=f"Evaluate a baseline detector ({_baseline_names})",
    )
    sp.add_argument("--baseline", required=True, help=f"Baseline name: {_baseline_names}")
    sp.add_argument("--model", type=Path, required=True, help="Path to trained weights")
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--split", default="test", choices=["train", "valid", "test"])
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default="0")
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--project", default="results")
    sp.add_argument("--name", default=None)
    sp.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))

    sp = sub.add_parser(
        "infer-baseline",
        help=f"Run inference with a baseline detector ({_baseline_names})",
    )
    sp.add_argument("--baseline", required=True, help=f"Baseline name: {_baseline_names}")
    sp.add_argument("--model", type=Path, required=True)
    sp.add_argument("--source", type=Path, required=True, help="Image file or folder")
    sp.add_argument("--out", type=Path, required=True, help="Output folder")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default="0")
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--save-images", action="store_true")
    sp.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))

    sp = sub.add_parser("list-baselines", help="List all available baseline detectors")

    # -------------------------------------------------------------------
    # Visibility head
    # -------------------------------------------------------------------

    sp = sub.add_parser(
        "train-visibility",
        help="Train the crop-based visibility classifier (good/medium/poor)",
    )
    sp.add_argument("--dataset", type=Path, default=_default_dataset_dir())
    sp.add_argument("--splits", nargs="+", default=["train"])
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--device", default="0")
    sp.add_argument("--out", type=Path, default=None, help="Output dir for model weights")

    sp = sub.add_parser(
        "benchmark",
        help="Run all baselines sequentially (train + eval) for comparative study",
    )
    sp.add_argument("--data", type=Path, default=_default_dataset_dir() / "data.yaml")
    sp.add_argument("--baselines", nargs="+", default=None, help="Subset of baselines (default: all)")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default="0")
    sp.add_argument("--project", default="results")
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--out", type=Path, default=Path("benchmark_results.md"))

    return p


def _parse_device(device_str: str):
    # Keep ultralytics-friendly device strings.
    s = str(device_str).strip()
    if s.isdigit():
        return int(s)
    return s


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "stats":
        print_split_counts(args.dataset, args.splits)
        return 0

    if args.cmd == "class-counts":
        print_class_instance_counts(args.dataset, args.splits, CLASS_NAMES, limit=args.limit)
        return 0

    if args.cmd == "checks":
        checks()
        return 0

    if args.cmd == "cuda-info":
        print_cuda_info()
        return 0

    if args.cmd == "train":
        uw = UnderwaterTrainConfig(
            use_udino=bool(args.underwater_udino),
            udino_radii=tuple(float(x) for x in args.udino_radii),
            udino_gate_reduction=int(args.udino_gate_reduction),
            use_fwnwd=bool(args.underwater_fwnwd),
            fwnwd_focal_gamma=float(args.fwnwd_gamma),
            fwnwd_wiou_distance_weight=float(args.fwnwd_wiou_k),
            fwnwd_nwd_lambda=float(args.fwnwd_nwd_lambda),
            fwnwd_normalizer=float(args.fwnwd_normalizer),
        )

        # Optional: apply visibility auxiliary loss
        if args.visibility_head:
            from fish_monitoring.training.visibility import apply_visibility_loss_patch
            data_yaml_path = _resolve_data_yaml(args.data)
            dataset_dir = data_yaml_path.parent
            apply_visibility_loss_patch(
                dataset_dir=dataset_dir,
                split="train",
                loss_weight=float(args.vis_loss_weight),
            )

        t = TrainArgs(
            data_yaml=_resolve_data_yaml(args.data),
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=_parse_device(args.device),
            fraction=float(args.fraction),
            val=not args.no_val,
            patience=args.patience,
            project=args.project,
            name=args.name,
            underwater=uw,
        )
        train(t, traceback_path=args.traceback_path)
        return 0

    if args.cmd == "train-fallback":
        uw = UnderwaterTrainConfig(
            use_udino=bool(args.underwater_udino),
            udino_radii=tuple(float(x) for x in args.udino_radii),
            udino_gate_reduction=int(args.udino_gate_reduction),
            use_fwnwd=bool(args.underwater_fwnwd),
            fwnwd_focal_gamma=float(args.fwnwd_gamma),
            fwnwd_wiou_distance_weight=float(args.fwnwd_wiou_k),
            fwnwd_nwd_lambda=float(args.fwnwd_nwd_lambda),
            fwnwd_normalizer=float(args.fwnwd_normalizer),
        )
        primary = TrainArgs(
            data_yaml=_resolve_data_yaml(args.data),
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=_parse_device(args.primary_device),
            fraction=float(args.fraction),
            val=True,
            patience=args.patience,
            project=args.project,
            name=args.primary_name,
            underwater=uw,
        )
        fallback = TrainArgs(
            data_yaml=_resolve_data_yaml(args.data),
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=_parse_device(args.fallback_device),
            fraction=float(args.fraction),
            val=True,
            patience=args.patience,
            project=args.project,
            name=args.fallback_name,
            underwater=uw,
        )
        train_with_fallback(primary, fallback, traceback_path=args.traceback_path)
        return 0

    if args.cmd == "eval":
        # 1) Ultralytics val (standard mAP metrics)
        v = ValArgs(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
        )
        # NOTE: yolo_ops.validate currently doesn't expose device, so we pass it via ultralytics directly here.
        # (keeps backward compatibility with ValArgs dataclass).
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(v.model_path))
        model.val(
            data=str(_resolve_data_yaml(v.data_yaml)),
            split=v.split,
            imgsz=v.imgsz,
            device=_parse_device(args.device),
            save_json=v.save_json,
            plots=v.plots,
            verbose=v.verbose,
            project=v.project,
            name=v.name,
        )

        # 2) Optional: export predictions + write our benchmark report
        if args.kitti_report is not None:
            from fish_monitoring.training.inference import InferConfig, run_final_inference
            from fish_monitoring.eval.report import ReportConfig, write_performance_report

            data_yaml = _resolve_data_yaml(args.data)
            dataset_dir = data_yaml.parent

            pred_out: Path = args.pred_out if args.pred_out is not None else (Path(args.project) / f"{args.name}_preds")
            source_dir = dataset_dir / str(args.split) / "images"

            run_final_inference(
                InferConfig(
                    model_path=Path(args.model),
                    source=source_dir,
                    output_dir=pred_out,
                    imgsz=int(args.imgsz),
                    iou=float(args.iou),
                    device=_parse_device(args.device),
                    conf=float(args.conf),
                    tiled=False,
                    tile_size=1024,
                    tile_overlap=0.25,
                    calib_dataset_dir=dataset_dir,
                    calib_split=str(args.split),
                    calib_max_images=None,
                    calib_min_precision=None,
                    calib_min_recall=None,
                    calib_mode="full",
                    calib_save_csv=None,
                    export_metrics=False,
                    metrics_dataset_dir=None,
                    metrics_split=str(args.split),
                    pr_conf=0.001,
                    metrics_num_classes=None,
                ),
                save_txt=True,
                save_images=False,
            )

            write_performance_report(
                ReportConfig(
                    dataset_dir=dataset_dir,
                    split=str(args.split),
                    pred_dir=(pred_out / "labels" if (pred_out / "labels").is_dir() else pred_out),
                    out_md=Path(args.kitti_report),
                    conf_th=float(args.conf),
                    iou_th=float(args.iou),
                    easy_min_h=(int(args.easy_min_h) if args.easy_min_h is not None else None),
                    moderate_min_h=(int(args.moderate_min_h) if args.moderate_min_h is not None else None),
                    hard_min_h=(int(args.hard_min_h) if args.hard_min_h is not None else None),
                    num_classes=(int(args.num_classes) if args.num_classes is not None else None),
                    size_pct=(float(args.size_pct) if args.size_pct is not None else None),
                    visibility_pct=(float(args.visibility_pct) if args.visibility_pct is not None else None),
                    background_pct=(float(args.background_pct) if args.background_pct is not None else None),
                )
            )
            print(f"Wrote report: {args.kitti_report}")
        return 0

    if args.cmd == "filter-labels":
        resize = AreaResize(
            original_width=args.original_width,
            original_height=args.original_height,
            new_width=args.new_width,
            new_height=args.new_height,
        )
        thresh = adjusted_area_threshold(resize, args.original_area_thresh)
        print(f"Adjusted area threshold: {thresh} px^2")

        results = filter_dataset_splits_by_area(
            dataset_base_dir=args.dataset,
            splits=list(args.splits),
            image_width=args.new_width,
            image_height=args.new_height,
            threshold_px2=thresh,
        )

        for split, (removed, total) in results.items():
            print(f"\n{split.upper()}: removed {total} boxes smaller than {thresh} px^2")
            for cls_id in sorted(removed.keys()):
                examples = removed[cls_id][:5]
                print(f"  class {cls_id}: {len(removed[cls_id])} removed")
                for fname, area in examples:
                    print(f"    - {fname} -> {area} px^2")
        return 0

    if args.cmd == "video":
        out = run_video(
            VideoArgs(
                weights=args.weights,
                input_video=args.input,
                output_dir=args.out,
                conf_thr=args.conf,
                tracker_cfg=args.tracker,
                seconds=args.seconds,
                device=_parse_device(args.device),
                crf=args.crf,
                preset=args.preset,
            )
        )
        print(f"Saved: {out}")
        return 0

    if args.cmd == "diagnose":
        from ultralytics import YOLO

        model = YOLO(str(args.model))
        r = diagnose(
            model=model,
            dataset_dir=args.dataset,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=_parse_device(args.device),
            tiled=args.tiled,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_images=args.max_images,
        )

        mode = "TILED" if args.tiled else "STANDARD"
        print(f"Mode: {mode}")
        print(f"Images: {r.images}")
        print(f"GT boxes: {r.gt_boxes}")
        print(f"Pred boxes: {r.pred_boxes}")
        print(f"TP/FP/FN: {r.tp}/{r.fp}/{r.fn}")
        print(f"Precision: {r.precision:.4f}")
        print(f"Recall:    {r.recall:.4f}")
        print("\nRecall by GT size bin:")
        for k, v in r.size_bins.items():
            print(f"  {k:<6}: recall={v['recall']:.3f} (gt={int(v['gt'])}, tp={int(v['tp'])}, fn={int(v['fn'])})")

        return 0

    if args.cmd == "infer":
        conf: float | None
        if str(args.conf).strip().lower() == "auto":
            conf = None
        else:
            conf = float(args.conf)

        cfg = InferConfig(
            model_path=args.model,
            source=args.source,
            output_dir=args.out,
            imgsz=args.imgsz,
            iou=args.iou,
            device=_parse_device(args.device),
            conf=conf,
            tiled=bool(args.tiled),
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            calib_dataset_dir=args.calib_dataset,
            calib_split=args.calib_split,
            calib_max_images=args.calib_max_images,
            calib_min_precision=args.min_precision,
            calib_min_recall=args.min_recall,
            calib_mode=args.calib_mode,
            calib_save_csv=args.calib_save_csv,
            export_metrics=bool(args.export_metrics),
            metrics_dataset_dir=args.metrics_dataset,
            metrics_split=str(args.metrics_split),
            pr_conf=float(args.pr_conf),
            metrics_num_classes=args.metrics_num_classes,
        )

        run_final_inference(cfg, save_txt=True, save_images=bool(args.save_images))
        print(f"Done. Outputs in: {args.out}")
        return 0

    if args.cmd == "report":
        from fish_monitoring.eval.report import ReportConfig, write_performance_report

        class_names: list[str] | None = None
        try:
            # Prefer dataset YAML names if present (falls back if PyYAML isn't installed)
            import yaml  # type: ignore

            data_yaml = args.dataset / "data.yaml"
            if data_yaml.exists():
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                names = data.get("names")
                if isinstance(names, list):
                    class_names = [str(x) for x in names]
        except Exception:
            class_names = None

        out_path = write_performance_report(
            ReportConfig(
                dataset_dir=args.dataset,
                split=args.split,
                pred_dir=args.pred,
                out_md=args.out,
                conf_th=float(args.conf),
                iou_th=float(args.iou),
                easy_min_h=(int(args.easy_min_h) if args.easy_min_h is not None else None),
                moderate_min_h=(int(args.moderate_min_h) if args.moderate_min_h is not None else None),
                hard_min_h=(int(args.hard_min_h) if args.hard_min_h is not None else None),
                num_classes=(int(args.num_classes) if args.num_classes is not None else None),
                size_pct=(float(args.size_pct) if args.size_pct is not None else None),
                visibility_pct=(float(args.visibility_pct) if args.visibility_pct is not None else None),
                background_pct=(float(args.background_pct) if args.background_pct is not None else None),
            ),
            class_names=class_names,
        )
        print(f"Wrote report: {out_path}")
        return 0

    if args.cmd == "spatial-errors":
        from fish_monitoring.eval.spatial import SpatialReportConfig, run_spatial_error_report

        summary_csv, fp_png, fn_png = run_spatial_error_report(
            SpatialReportConfig(
                dataset_dir=args.dataset,
                split=str(args.split),
                pred_labels_dir=args.pred_labels,
                out_dir=args.out,
                conf_th=float(args.conf),
                iou_th=float(args.iou),
                min_height_px=int(args.min_height),
                grid_w=int(args.grid_w),
                grid_h=int(args.grid_h),
                export_points=bool(args.export_points),
            )
        )
        print(f"Wrote: {summary_csv}")
        if fp_png is not None:
            print(f"Wrote: {fp_png}")
        if fn_png is not None:
            print(f"Wrote: {fn_png}")
        return 0

    if args.cmd == "underwater":
        # Keep this import lazy so torch isn't required for other commands.
        from fish_monitoring.underwater.demo import main as uw_main

        forwarded = list(args.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return int(uw_main(forwarded))

    # -------------------------------------------------------------------
    # Baseline detectors
    # -------------------------------------------------------------------

    if args.cmd == "list-baselines":
        from fish_monitoring.baselines.registry import list_baselines
        print("Available baseline detectors:")
        for name in list_baselines():
            print(f"  - {name}")
        return 0

    if args.cmd == "train-baseline":
        from fish_monitoring.baselines import get_baseline
        from fish_monitoring.baselines.base_detector import BaselineTrainConfig

        detector = get_baseline(args.baseline)
        run_name = args.name or f"{detector.name}_train"
        cfg = BaselineTrainConfig(
            data_yaml=_resolve_data_yaml(args.data),
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=_parse_device(args.device),
            patience=args.patience,
            project=args.project,
            name=run_name,
            lr=args.lr,
            num_classes=args.num_classes,
            class_names=list(CLASS_NAMES[:args.num_classes]),
        )
        best_path = detector.train(cfg)
        print(f"[{detector.name}] Best weights saved to: {best_path}")
        return 0

    if args.cmd == "eval-baseline":
        from fish_monitoring.baselines import get_baseline
        from fish_monitoring.baselines.base_detector import BaselineEvalConfig

        detector = get_baseline(args.baseline)
        run_name = args.name or f"{detector.name}_eval"
        cfg = BaselineEvalConfig(
            model_path=args.model,
            data_yaml=_resolve_data_yaml(args.data),
            split=args.split,
            imgsz=args.imgsz,
            device=_parse_device(args.device),
            conf=args.conf,
            iou=args.iou,
            project=args.project,
            name=run_name,
            num_classes=args.num_classes,
            class_names=list(CLASS_NAMES[:args.num_classes]),
        )
        metrics = detector.evaluate(cfg)
        print(f"\n[{detector.name}] Final metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "infer-baseline":
        from fish_monitoring.baselines import get_baseline
        from fish_monitoring.baselines.base_detector import BaselineInferConfig

        detector = get_baseline(args.baseline)
        cfg = BaselineInferConfig(
            model_path=args.model,
            source=args.source,
            output_dir=args.out,
            imgsz=args.imgsz,
            device=_parse_device(args.device),
            conf=args.conf,
            iou=args.iou,
            save_txt=True,
            save_images=bool(args.save_images),
            num_classes=args.num_classes,
            class_names=list(CLASS_NAMES[:args.num_classes]),
        )
        detector.infer_directory(cfg)
        print(f"[{detector.name}] Inference outputs: {args.out}")
        return 0

    if args.cmd == "train-visibility":
        from fish_monitoring.training.visibility import train_crop_visibility_classifier

        out = train_crop_visibility_classifier(
            dataset_dir=args.dataset,
            splits=list(args.splits),
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=_parse_device(args.device),
            output_dir=args.out,
        )
        print(f"Visibility classifier saved to: {out}")
        return 0

    if args.cmd == "benchmark":
        from fish_monitoring.baselines import get_baseline
        from fish_monitoring.baselines.registry import list_baselines
        from fish_monitoring.baselines.base_detector import BaselineTrainConfig, BaselineEvalConfig

        available = list_baselines()
        selected = args.baselines if args.baselines else available
        invalid = [b for b in selected if b not in available]
        if invalid:
            print(f"Unknown baselines: {invalid}. Available: {available}")
            return 1

        results_table: list[dict[str, Any]] = []
        data_yaml = _resolve_data_yaml(args.data)

        for baseline_name in selected:
            print(f"\n{'='*60}")
            print(f"  BASELINE: {baseline_name}")
            print(f"{'='*60}\n")

            try:
                detector = get_baseline(baseline_name)

                # Train
                train_cfg = BaselineTrainConfig(
                    data_yaml=data_yaml,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=_parse_device(args.device),
                    project=args.project,
                    name=f"{baseline_name}_benchmark",
                )
                best_path = detector.train(train_cfg)

                # Evaluate
                eval_cfg = BaselineEvalConfig(
                    model_path=best_path,
                    data_yaml=data_yaml,
                    split="test",
                    imgsz=args.imgsz,
                    device=_parse_device(args.device),
                    conf=args.conf,
                    iou=args.iou,
                    project=args.project,
                    name=f"{baseline_name}_benchmark_eval",
                )
                metrics = detector.evaluate(eval_cfg)
                metrics["baseline"] = baseline_name
                metrics["weights"] = str(best_path)
                results_table.append(metrics)

            except Exception as e:
                print(f"[{baseline_name}] FAILED: {e}")
                results_table.append({"baseline": baseline_name, "error": str(e)})

        # Write benchmark summary
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        md_lines = [
            "# Baseline Benchmark Results\n\n",
            f"Generated: `{now}`\n\n",
            f"Dataset: `{data_yaml}`\n\n",
            "## Results\n\n",
            "| Baseline | mAP50 | mAP50-95 | Precision | Recall | F1 | Status |\n",
            "|----------|-------|----------|-----------|--------|----|--------|\n",
        ]

        for r in results_table:
            name = r.get("baseline", "?")
            if "error" in r:
                md_lines.append(f"| {name} | — | — | — | — | — | ❌ {r['error'][:50]} |\n")
            else:
                m50 = f"{r.get('mAP50', 0):.4f}" if "mAP50" in r else "—"
                m5095 = f"{r.get('mAP50-95', 0):.4f}" if "mAP50-95" in r else "—"
                p = f"{r.get('precision', 0):.4f}"
                rc = f"{r.get('recall', 0):.4f}"
                f1 = f"{r.get('f1', 0):.4f}" if "f1" in r else "—"
                md_lines.append(f"| {name} | {m50} | {m5095} | {p} | {rc} | {f1} | ✅ |\n")

        md_lines.append("\n")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(md_lines), encoding="utf-8")
        print(f"\nBenchmark summary: {args.out}")
        return 0

    raise AssertionError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
