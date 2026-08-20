"""YOLO-NAS baseline via Deci's super-gradients.

YOLO-NAS (Neural Architecture Search) was designed by Deci using AutoNAC
(Automated Neural Architecture Construction). It often outperforms YOLOv8
on standard benchmarks while maintaining fast inference.

Usage:
    python main.py train-baseline --baseline yolo-nas \
        --data ../data/WIO-ReefFish/data.yaml \
        --weights yolo_nas_s --epochs 100 --batch 16
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from fish_monitoring.baselines.base_detector import (
    BaseDetector,
    BaselineEvalConfig,
    BaselineInferConfig,
    BaselineTrainConfig,
    iter_split_images,
    parse_data_yaml,
    read_yolo_labels,
    resolve_split_dirs,
)
from fish_monitoring.constants import CLASS_NAMES
from fish_monitoring.core.inference import Pred


def _import_super_gradients():
    try:
        import super_gradients
        return super_gradients
    except ImportError as e:
        raise RuntimeError(
            "super-gradients is required for YOLO-NAS. Install it with:\n"
            "  pip install super-gradients"
        ) from e


class YOLONASDetector(BaseDetector):
    """YOLO-NAS detector via Deci super-gradients."""

    name = "yolo-nas"

    def _resolve_dataset_params(self, data_yaml: Path, imgsz: int):
        """Build super-gradients dataset params from YOLO data.yaml."""
        sg = _import_super_gradients()
        cfg = parse_data_yaml(data_yaml)
        dataset_dir = data_yaml.parent

        # Resolve path
        root = cfg.get("path", str(dataset_dir))
        if not Path(root).is_absolute():
            root = str(dataset_dir / root)

        class_names = cfg.get("names", [])
        nc = cfg.get("nc", len(class_names))

        return root, class_names, nc

    def train(self, cfg: BaselineTrainConfig) -> Path:
        sg = _import_super_gradients()
        from super_gradients.training import models, dataloaders, Trainer
        from super_gradients.training.losses import PPYoloELoss
        from super_gradients.training.metrics import DetectionMetrics_050

        root, class_names, nc = self._resolve_dataset_params(cfg.data_yaml, cfg.imgsz)

        if not class_names:
            from fish_monitoring.constants import CLASS_NAMES
            class_names = list(CLASS_NAMES)
            nc = len(class_names)

        # Model
        model_name = cfg.weights or "yolo_nas_s"
        model = models.get(
            model_name,
            num_classes=nc,
            pretrained_weights="coco" if "yolo_nas" in model_name else None,
        )

        # Dataset setup (YOLO format)
        train_data = dataloaders.get(
            name="coco_detection_yolo_format_train",
            dataset_params={
                "data_dir": root,
                "images_dir": "train/images",
                "labels_dir": "train/labels",
                "classes": class_names,
                "input_dim": [cfg.imgsz, cfg.imgsz],
            },
            dataloader_params={
                "batch_size": cfg.batch,
                "num_workers": min(4, os.cpu_count() or 1),
            },
        )

        val_data = dataloaders.get(
            name="coco_detection_yolo_format_val",
            dataset_params={
                "data_dir": root,
                "images_dir": "valid/images",
                "labels_dir": "valid/labels",
                "classes": class_names,
                "input_dim": [cfg.imgsz, cfg.imgsz],
            },
            dataloader_params={
                "batch_size": cfg.batch,
                "num_workers": min(4, os.cpu_count() or 1),
            },
        )

        # Training params
        train_params = {
            "max_epochs": cfg.epochs,
            "lr_mode": "cosine",
            "initial_lr": cfg.lr,
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 1e-4},
            "loss": PPYoloELoss(
                num_classes=nc,
                use_static_assigner=False,
                reg_max=16,
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=cfg.batch,
                    top_k_predictions=300,
                    num_cls=nc,
                    normalize_targets=True,
                    post_prediction_callback=None,
                )
            ],
            "metric_to_watch": "mAP@0.50",
            "early_stopping_patience": cfg.patience,
            "save_ckpt_epoch_list": [],
            "average_best_models": False,
        }

        # Use device
        trainer = Trainer(
            experiment_name=cfg.name,
            ckpt_root_dir=cfg.project,
        )

        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data,
        )

        best = Path(cfg.project) / cfg.name / "ckpt_best.pth"
        print(f"[YOLO-NAS] Training complete. Best weights: {best}")
        return best

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        sg = _import_super_gradients()
        from super_gradients.training import models

        root, class_names, nc = self._resolve_dataset_params(cfg.data_yaml, cfg.imgsz)

        if not class_names:
            from fish_monitoring.constants import CLASS_NAMES
            class_names = list(CLASS_NAMES)
            nc = len(class_names)

        model = models.get(
            "yolo_nas_s",
            num_classes=nc,
            checkpoint_path=str(cfg.model_path),
        )

        # Run inference on each image and evaluate
        from fish_monitoring.eval.diagnose import _match_predictions, Gt, _load_image_size

        _, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            pred = self._predict_with_model(model, img_path, cfg.imgsz, cfg.conf)
            w, h = _load_image_size(img_path)
            gt_xyxy, gt_cls = read_yolo_labels(labels_dir / f"{img_path.stem}.txt", im_w=w, im_h=h)
            gt = Gt(xyxy=gt_xyxy, cls=gt_cls)
            tp_i, fp_i, fn_i, _ = _match_predictions(gt, pred, iou_th=cfg.iou)
            tp_total += tp_i
            fp_total += fp_i
            fn_total += fn_i

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {"precision": prec, "recall": rec, "f1": f1, "tp": tp_total, "fp": fp_total, "fn": fn_total}
        print(f"[YOLO-NAS] Eval: {metrics}")
        return metrics

    def _predict_with_model(self, model, image_path: Path, imgsz: int, conf: float) -> Pred:
        """Run prediction using super-gradients model."""
        results = model.predict(str(image_path), conf=conf, fuse_model=False)

        # Extract predictions
        pred = results._images_prediction_lst[0]
        bboxes = pred.prediction.bboxes_xyxy  # (N, 4)
        confidence = pred.prediction.confidence  # (N,)
        labels = pred.prediction.labels  # (N,)

        if bboxes.shape[0] == 0:
            return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                        conf=np.zeros((0,), dtype=np.float32),
                        cls=np.zeros((0,), dtype=np.int64))

        return Pred(
            xyxy=bboxes.astype(np.float32),
            conf=confidence.astype(np.float32),
            cls=labels.astype(np.int64),
        )

    def predict(
        self, image_path: Path, *, model_path: Path,
        imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: Any = 0,
    ) -> Pred:
        sg = _import_super_gradients()
        from super_gradients.training import models

        if not hasattr(self, "_ynas_model") or self._ynas_path != str(model_path):
            nc = len(CLASS_NAMES)
            # Try to load from checkpoint
            self._ynas_model = models.get(
                "yolo_nas_s",
                num_classes=nc,
                checkpoint_path=str(model_path),
            )
            self._ynas_path = str(model_path)

        return self._predict_with_model(self._ynas_model, image_path, imgsz, conf)
