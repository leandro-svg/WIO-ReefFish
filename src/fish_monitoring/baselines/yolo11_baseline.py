"""YOLOv11 (YOLO11) baseline via Ultralytics.

YOLO11 is the latest generation in the YOLO family (Ultralytics, 2024).
It introduces architectural improvements including C3k2 blocks and
attention mechanisms while maintaining real-time inference.

The user referred to this as "yolo-26" — we interpret this as the latest
YOLO version available in Ultralytics (YOLO11 / YOLOv11).

Usage:
    python main.py train-baseline --baseline yolo11 \
        --data ../data/WIO-ReefFish/data.yaml \
        --weights yolo11m.pt --epochs 100
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


class YOLO11Detector(BaseDetector):
    """YOLOv11 (latest Ultralytics YOLO) baseline."""

    name = "yolo11"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        from ultralytics import YOLO

        weights = cfg.weights or "yolo11m.pt"
        model = YOLO(weights)

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
        print(f"[YOLO11] Training complete. Best weights: {best}")
        return best

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        from ultralytics import YOLO

        model = YOLO(str(cfg.model_path))

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
        print(f"[YOLO11] Eval: {metrics}")
        return metrics

    def predict(
        self, image_path: Path, *, model_path: Path,
        imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: Any = 0,
    ) -> Pred:
        from ultralytics import YOLO

        if not hasattr(self, "_model") or self._y11_path != str(model_path):
            self._model = YOLO(str(model_path))
            self._y11_path = str(model_path)

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
