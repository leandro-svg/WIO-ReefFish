"""RetinaNet baseline via torchvision.

RetinaNet (Lin et al., 2017) introduced focal loss to address extreme
foreground-background class imbalance in dense single-stage detectors.
We use ``retinanet_resnet50_fpn_v2`` from torchvision, following the same
dataset adapter pattern as the Faster R-CNN baseline.

Usage:
    python main.py train-baseline --baseline retinanet \
        --data ../data/WIO-ReefFish/data.yaml \
        --epochs 50 --batch 8 --lr 0.005
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
    read_yolo_labels,
    resolve_split_dirs,
)
from fish_monitoring.constants import CLASS_NAMES
from fish_monitoring.core.inference import Pred


# Re-use the same YOLO→torchvision dataset adapter from Faster R-CNN
from fish_monitoring.baselines.faster_rcnn_baseline import (
    _YOLODetectionDataset,
    _collate_fn,
)


class RetinaNetDetector(BaseDetector):
    """RetinaNet with ResNet-50 FPN v2 backbone (torchvision)."""

    name = "retinanet"

    def _build_model(self, num_classes: int, pretrained_backbone: bool = True):
        import torch
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        # torchvision RetinaNet expects num_classes *including* background
        model = retinanet_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained_backbone else None,
        )

        # Replace classification head for our number of classes
        # RetinaNet: num_classes includes background (index 0)
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels

        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,  # +1 for background
        )

        # Also need to update the score_thresh on the model to match our conf
        model.score_thresh = 0.05  # low during training; we filter at eval

        return model

    # ── training ─────────────────────────────────────────────────────────
    def train(self, cfg: BaselineTrainConfig) -> Path:
        import torch
        from torch.utils.data import DataLoader

        device = torch.device(
            f"cuda:{cfg.device}"
            if isinstance(cfg.device, int) and torch.cuda.is_available()
            else "cpu"
        )

        model = self._build_model(cfg.num_classes)
        model.to(device)

        train_ds = _YOLODetectionDataset(cfg.data_yaml, "train", imgsz=cfg.imgsz)
        valid_ds = _YOLODetectionDataset(cfg.data_yaml, "valid", imgsz=cfg.imgsz)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            collate_fn=_collate_fn,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            collate_fn=_collate_fn,
            pin_memory=True,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        out_dir = Path(cfg.project) / cfg.name / "weights"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # ── Training ──
            model.train()
            epoch_loss = 0.0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                epoch_loss += float(losses)

            lr_scheduler.step()

            # ── Validation loss ──
            model.train()  # RetinaNet returns losses only in train mode
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in valid_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    try:
                        loss_dict = model(images, targets)
                        val_loss += float(sum(loss for loss in loss_dict.values()))
                    except Exception:
                        pass

            avg_train = epoch_loss / max(len(train_loader), 1)
            avg_val = val_loss / max(len(valid_loader), 1)
            print(
                f"[RetinaNet] Epoch {epoch + 1}/{cfg.epochs}  "
                f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
            )

            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(model.state_dict(), out_dir / "last.pt")

            if patience_counter >= cfg.patience:
                print(f"[RetinaNet] Early stopping at epoch {epoch + 1}")
                break

        # Save metadata for inference
        meta = {
            "num_classes": cfg.num_classes,
            "class_names": cfg.class_names,
            "imgsz": cfg.imgsz,
        }
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[RetinaNet] Training complete. Best weights: {best_path}")
        return best_path

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        import torch
        from fish_monitoring.eval.diagnose import Gt, _load_image_size, _match_predictions

        device = torch.device(
            f"cuda:{cfg.device}"
            if isinstance(cfg.device, int) and torch.cuda.is_available()
            else "cpu"
        )

        model = self._build_model(cfg.num_classes)
        model.load_state_dict(
            torch.load(str(cfg.model_path), map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        _, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            pred = self._predict_with_model(model, img_path, device, cfg.imgsz, cfg.conf)
            w, h = _load_image_size(img_path)

            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)
            gt = Gt(xyxy=gt_xyxy, cls=gt_cls)

            tp_i, fp_i, fn_i, _ = _match_predictions(gt, pred, iou_th=cfg.iou)
            tp_total += tp_i
            fp_total += fp_i
            fn_total += fn_i

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        }
        print(f"[RetinaNet] Eval: {metrics}")
        return metrics

    # ── internal predict helper ──────────────────────────────────────────
    def _predict_with_model(
        self, model, image_path: Path, device, imgsz: int, conf: float
    ) -> Pred:
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((imgsz, imgsz))
        img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # Filter by confidence
        keep = scores >= conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] == 0:
            return Pred(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int64),
            )

        # Scale back to original image size
        sx = orig_w / imgsz
        sy = orig_h / imgsz
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        # Convert 1-indexed → 0-indexed labels
        labels = labels - 1

        return Pred(
            xyxy=boxes.astype(np.float32),
            conf=scores.astype(np.float32),
            cls=labels.astype(np.int64),
        )

    # ── public predict (with caching) ────────────────────────────────────
    def predict(
        self,
        image_path: Path,
        *,
        model_path: Path,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        device: Any = 0,
    ) -> Pred:
        import torch

        dev = torch.device(
            f"cuda:{device}"
            if isinstance(device, int) and torch.cuda.is_available()
            else "cpu"
        )

        if not hasattr(self, "_model") or self._model_path_str != str(model_path):
            meta_path = model_path.parent / "meta.pt"
            nc = len(CLASS_NAMES)
            if meta_path.exists():
                meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
                nc = meta.get("num_classes", len(CLASS_NAMES))

            self._retina_model = self._build_model(nc)
            self._retina_model.load_state_dict(
                torch.load(str(model_path), map_location=dev, weights_only=True)
            )
            self._retina_model.to(dev)
            self._retina_model.eval()
            self._model_path_str = str(model_path)
            self._retina_device = dev

        return self._predict_with_model(
            self._retina_model, image_path, self._retina_device, imgsz, conf
        )
