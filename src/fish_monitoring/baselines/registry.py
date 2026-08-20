"""Baseline registry — maps short names to detector classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fish_monitoring.baselines.base_detector import BaseDetector


# Lazy registry: maps name -> (module_path, class_name)
_LAZY_REGISTRY: dict[str, tuple[str, str]] = {
    "rtdetr": ("fish_monitoring.baselines.rtdetr_baseline", "RTDETRDetector"),
    "faster-rcnn": ("fish_monitoring.baselines.faster_rcnn_baseline", "FasterRCNNDetector"),
    "dinov2": ("fish_monitoring.baselines.dinov2_backbone", "DINOv2Detector"),
    "dinov2-frcnn": ("fish_monitoring.baselines.dinov2_frcnn", "DINOv2FasterRCNNDetector"),
    "sam2": ("fish_monitoring.baselines.sam2_baseline", "SAM2Detector"),
    "yolo-world": ("fish_monitoring.baselines.yolo_world_baseline", "YOLOWorldDetector"),
    "yolo11": ("fish_monitoring.baselines.yolo11_baseline", "YOLO11Detector"),
    "yolo-nas": ("fish_monitoring.baselines.yolo_nas_baseline", "YOLONASDetector"),
    "yolov8": ("fish_monitoring.baselines.yolov8_baseline", "YOLOv8Detector"),
    "retinanet": ("fish_monitoring.baselines.retinanet_baseline", "RetinaNetDetector"),
    "grounding-dino": ("fish_monitoring.baselines.grounding_dino_baseline", "GroundingDINODetector"),
    "co-detr": ("fish_monitoring.baselines.co_detr_baseline", "CoDETRDetector"),
    "yolo26": ("fish_monitoring.baselines.yolo26_baseline", "YOLO26Detector"),
}


def _load_class(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# Populated on first access
BASELINE_REGISTRY: dict[str, type] = {}


def get_baseline(name: str) -> "BaseDetector":
    """Instantiate a baseline detector by short name."""
    key = name.lower().strip()
    if key not in _LAZY_REGISTRY:
        available = ", ".join(sorted(_LAZY_REGISTRY.keys()))
        raise ValueError(f"Unknown baseline '{name}'. Available: {available}")

    if key not in BASELINE_REGISTRY:
        mod_path, cls_name = _LAZY_REGISTRY[key]
        BASELINE_REGISTRY[key] = _load_class(mod_path, cls_name)

    return BASELINE_REGISTRY[key]()


def list_baselines() -> list[str]:
    return sorted(_LAZY_REGISTRY.keys())
