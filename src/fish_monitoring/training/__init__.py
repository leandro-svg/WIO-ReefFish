"""Training: YOLO ops, final inference pipeline, video, visibility head."""

from fish_monitoring.training.yolo_ops import (
    TrainArgs, ValArgs, train, train_with_fallback, validate,
    checks, print_cuda_info,
)
from fish_monitoring.training.video import VideoArgs, run_video
from fish_monitoring.training.inference import InferConfig, run_final_inference

__all__ = [
    "TrainArgs", "ValArgs", "train", "train_with_fallback", "validate",
    "checks", "print_cuda_info",
    "VideoArgs", "run_video",
    "InferConfig", "run_final_inference",
]
