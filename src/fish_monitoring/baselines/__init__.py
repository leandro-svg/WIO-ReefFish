"""Baseline detection models for comparative evaluation.

Each baseline implements a common interface (BaseDetector) so that
training, evaluation, and inference follow the same CLI workflow as YOLO.

Supported baselines:
  - RT-DETR          (Ultralytics)
  - Faster R-CNN     (torchvision)
  - DINOv2 backbone  (facebookresearch/dinov2)
  - DINOv2 + Faster R-CNN
  - SAM 2            (prompted zero-shot / refinement)
  - YOLO-World       (open-vocabulary, Ultralytics)
  - YOLOv11          (latest Ultralytics YOLO)
  - YOLO-NAS         (super-gradients)
"""

from fish_monitoring.baselines.registry import BASELINE_REGISTRY, get_baseline

__all__ = ["BASELINE_REGISTRY", "get_baseline"]
