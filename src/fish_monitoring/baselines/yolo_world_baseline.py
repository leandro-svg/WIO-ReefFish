"""YOLO-World baseline (open-vocabulary detection).

YOLO-World extends YOLO with a text encoder for open-vocabulary detection.
For our fish dataset, we fine-tune it with our 24 fish family class names
as text prompts, leveraging its vision-language alignment.

Usage:
    python main.py train-baseline --baseline yolo-world \
        --data ../data/WIO-ReefFish/data.yaml \
        --weights yolov8s-world.pt --epochs 100
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fish_monitoring.baselines.base_detector import (
    BaseDetector,
    BaselineEvalConfig,
    BaselineInferConfig,
    BaselineTrainConfig,
)
from fish_monitoring.core.inference import Pred


class YOLOWorldDetector(BaseDetector):
    """YOLO-World open-vocabulary detection, fine-tuned on fish classes."""

    name = "yolo-world"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        from ultralytics import YOLOWorld

        weights = cfg.weights or "yolov8s-world.pt"
        model = YOLOWorld(weights)

        # Set fish class names as text prompts
        model.set_classes(cfg.class_names)

        model.train(
            data=str(cfg.data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            patience=cfg.patience,
            project=cfg.project,
            name=cfg.name,
            lr0=cfg.lr,
        )

        best = Path(cfg.project) / cfg.name / "weights" / "best.pt"
        print(f"[YOLO-World] Training complete. Best weights: {best}")
        return best

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        from ultralytics import YOLOWorld

        model = YOLOWorld(str(cfg.model_path))
        model.set_classes(cfg.class_names)

        # ultralytics uses 'val' key from data.yaml, not 'valid'
        ul_split = "val" if cfg.split == "valid" else cfg.split
        results = model.val(
            data=str(cfg.data_yaml),
            split=ul_split,
            imgsz=cfg.imgsz,
            device=cfg.device,
            conf=cfg.conf,
            iou=cfg.iou,
            project=cfg.project,
            name=cfg.name,
        )

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        print(f"[YOLO-World] Eval: {metrics}")
        return metrics

    def predict(
        self, image_path: Path, *, model_path: Path,
        imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: Any = 0,
    ) -> Pred:
        from ultralytics import YOLOWorld

        if not hasattr(self, "_model") or self._yw_path != str(model_path):
            self._model = YOLOWorld(str(model_path))
            from fish_monitoring.constants import CLASS_NAMES
            self._model.set_classes(list(CLASS_NAMES))
            self._yw_path = str(model_path)

        results = self._model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                        conf=np.zeros((0,), dtype=np.float32),
                        cls=np.zeros((0,), dtype=np.int64))

        return Pred(
            xyxy=boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            conf=boxes.conf.detach().cpu().numpy().astype(np.float32),
            cls=boxes.cls.detach().cpu().numpy().astype(np.int64),
        )
