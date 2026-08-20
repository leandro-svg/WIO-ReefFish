"""DINOv2 + Faster R-CNN baseline.

Replaces the standard ResNet-50 backbone of Faster R-CNN with frozen DINOv2
features, combining powerful self-supervised representations with the mature
Faster R-CNN detection framework.

Usage:
    python main.py train-baseline --baseline dinov2-frcnn \
        --data ../data/WIO-ReefFish/data.yaml \
        --epochs 50 --batch 4 --lr 0.001
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, OrderedDict

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


# ---------------------------------------------------------------------------
# DINOv2 backbone wrapper for torchvision Faster R-CNN
# ---------------------------------------------------------------------------

def _build_dinov2_frcnn(num_classes: int, backbone_size: str = "small"):
    """Faster R-CNN with DINOv2 backbone instead of ResNet.

    The DINOv2 ViT is frozen and its patch features are reshaped into
    multi-scale feature maps via a simple FPN-like neck, then plugged
    into torchvision's Faster R-CNN framework.
    """
    import torch
    import torch.nn as nn
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    model_map = {
        "small": ("dinov2_vits14", 384),
        "base": ("dinov2_vitb14", 768),
        "large": ("dinov2_vitl14", 1024),
    }
    model_name, feat_dim = model_map.get(backbone_size, model_map["small"])

    class DINOv2BackboneWithFPN(nn.Module):
        """Wraps DINOv2 as a backbone compatible with Faster R-CNN.

        Produces a dict of feature maps at different 'scales' by
        spatial pooling of the patch-token grid.
        """

        def __init__(self):
            super().__init__()
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)

            # Freeze everything first
            for p in self.dinov2.parameters():
                p.requires_grad = False

            # Unfreeze last 4 transformer blocks + final norm so features
            # can adapt from classification to detection / localization.
            n_blocks = len(self.dinov2.blocks)
            for blk in self.dinov2.blocks[max(0, n_blocks - 4):]:
                for p in blk.parameters():
                    p.requires_grad = True
            if hasattr(self.dinov2, 'norm'):
                for p in self.dinov2.norm.parameters():
                    p.requires_grad = True

            self.feat_dim = feat_dim
            self.patch_size = 14
            self.out_channels = 256

            # Project to common channel dim at multiple scales
            self.proj_0 = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.proj_1 = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.proj_2 = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(4, 4),
            )

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            B, C, H_in, W_in = x.shape
            H = (H_in // self.patch_size) * self.patch_size
            W = (W_in // self.patch_size) * self.patch_size
            if H != H_in or W != W_in:
                x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

            # No torch.no_grad() here — last 4 unfrozen blocks need gradients
            features = self.dinov2.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]

            h_p = H // self.patch_size
            w_p = W // self.patch_size
            feat_map = patch_tokens.transpose(1, 2).reshape(B, self.feat_dim, h_p, w_p)

            return OrderedDict([
                ("feat0", self.proj_0(feat_map)),
                ("feat1", self.proj_1(feat_map)),
                ("feat2", self.proj_2(feat_map)),
            ])

    backbone = DINOv2BackboneWithFPN()

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),) * 3,
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["feat0", "feat1", "feat2"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=224,
        max_size=800,
        # Disable internal normalization — our dataset already applies
        # ImageNet mean/std, which DINOv2 expects. Without this override,
        # images get double-normalised and DINOv2 receives garbage features.
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )

    return model


# ---------------------------------------------------------------------------
# Dataset adapter (reuse from faster_rcnn_baseline)
# ---------------------------------------------------------------------------

class _YOLODetectionDataset:
    def __init__(self, data_yaml: Path, split: str, imgsz: int = 518, augment: bool = False):
        self.images = iter_split_images(data_yaml, split)
        _, self.labels_dir = resolve_split_dirs(data_yaml, split)
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        import random

        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.imgsz, self.imgsz))

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        xyxy, cls = read_yolo_labels(label_path, im_w=orig_w, im_h=orig_h)

        # Data augmentation (training only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if xyxy.shape[0] > 0:
                    # Flip boxes: x1_new = orig_w - x2, x2_new = orig_w - x1
                    x1 = orig_w - xyxy[:, 2].copy()
                    x2 = orig_w - xyxy[:, 0].copy()
                    xyxy[:, 0] = x1
                    xyxy[:, 2] = x2

            # Color jitter
            from torchvision import transforms
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            img = jitter(img)

        img_tensor = F.to_tensor(img)

        # DINOv2 normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        if xyxy.shape[0] > 0:
            sx = self.imgsz / orig_w
            sy = self.imgsz / orig_h
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy
            labels = cls + 1  # 1-indexed
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def _collate_fn(batch):
    return tuple(zip(*batch))


class DINOv2FasterRCNNDetector(BaseDetector):
    """DINOv2 backbone + Faster R-CNN detection framework."""

    name = "dinov2-frcnn"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        import torch
        from torch.utils.data import DataLoader

        device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) and torch.cuda.is_available() else "cpu")
        imgsz = ((cfg.imgsz + 13) // 14) * 14

        model = _build_dinov2_frcnn(cfg.num_classes, backbone_size="small")
        model.to(device)

        train_ds = _YOLODetectionDataset(cfg.data_yaml, "train", imgsz=imgsz, augment=True)
        valid_ds = _YOLODetectionDataset(cfg.data_yaml, "valid", imgsz=imgsz, augment=False)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                                  num_workers=min(4, os.cpu_count() or 1), collate_fn=_collate_fn, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=cfg.batch, shuffle=False,
                                  num_workers=min(4, os.cpu_count() or 1), collate_fn=_collate_fn, pin_memory=True)

        # Only train non-frozen params
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        out_dir = Path(cfg.project) / cfg.name / "weights"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            # The early (frozen) blocks stay in eval mode automatically
            # because their params have requires_grad=False.
            # The last 4 unfrozen blocks need train mode.
            epoch_loss = 0.0

            for images_batch, targets_batch in train_loader:
                images = [img.to(device) for img in images_batch]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
                optimizer.step()
                epoch_loss += float(losses)

            scheduler.step()

            # Validation
            model.train()
            val_loss = 0.0
            with torch.no_grad():
                for images_batch, targets_batch in valid_loader:
                    images = [img.to(device) for img in images_batch]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]
                    try:
                        loss_dict = model(images, targets)
                        val_loss += float(sum(loss for loss in loss_dict.values()))
                    except Exception:
                        pass

            avg_train = epoch_loss / max(len(train_loader), 1)
            avg_val = val_loss / max(len(valid_loader), 1)
            print(f"[DINOv2+FRCNN] Epoch {epoch + 1}/{cfg.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"[DINOv2+FRCNN] Early stopping at epoch {epoch + 1}")
                break

        meta = {"num_classes": cfg.num_classes, "class_names": cfg.class_names,
                "imgsz": imgsz, "backbone_size": "small"}
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[DINOv2+FRCNN] Training complete. Best weights: {best_path}")
        return best_path

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        import torch
        from fish_monitoring.eval.diagnose import _match_predictions, Gt, _load_image_size

        device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) and torch.cuda.is_available() else "cpu")

        meta_path = cfg.model_path.parent / "meta.pt"
        nc = cfg.num_classes
        imgsz = cfg.imgsz
        if meta_path.exists():
            meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
            nc = meta.get("num_classes", nc)
            imgsz = meta.get("imgsz", imgsz)

        imgsz = ((imgsz + 13) // 14) * 14

        model = _build_dinov2_frcnn(nc, "small")
        state = torch.load(str(cfg.model_path), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        _, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            pred = self._predict_with_model(model, img_path, device, imgsz, cfg.conf)
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
        print(f"[DINOv2+FRCNN] Eval: {metrics}")
        return metrics

    def _predict_with_model(self, model, image_path: Path, device, imgsz: int, conf: float) -> Pred:
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((imgsz, imgsz))
        img_tensor = F.to_tensor(img_resized)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            outputs = model([img_tensor.to(device)])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        keep = scores >= conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.shape[0] == 0:
            return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                        conf=np.zeros((0,), dtype=np.float32),
                        cls=np.zeros((0,), dtype=np.int64))

        sx = orig_w / imgsz
        sy = orig_h / imgsz
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        labels = labels - 1  # back to 0-indexed

        return Pred(xyxy=boxes.astype(np.float32), conf=scores.astype(np.float32), cls=labels.astype(np.int64))

    def predict(
        self, image_path: Path, *, model_path: Path,
        imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: Any = 0,
    ) -> Pred:
        import torch

        dev = torch.device(f"cuda:{device}" if isinstance(device, int) and torch.cuda.is_available() else "cpu")

        if not hasattr(self, "_d2frcnn_model") or self._d2frcnn_path != str(model_path):
            meta_path = model_path.parent / "meta.pt"
            nc = len(CLASS_NAMES)
            if meta_path.exists():
                meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
                nc = meta.get("num_classes", len(CLASS_NAMES))
                imgsz = meta.get("imgsz", imgsz)

            imgsz = ((imgsz + 13) // 14) * 14
            self._d2frcnn_model = _build_dinov2_frcnn(nc, "small")
            state = torch.load(str(model_path), map_location=dev, weights_only=True)
            self._d2frcnn_model.load_state_dict(state, strict=False)
            self._d2frcnn_model.to(dev)
            self._d2frcnn_model.eval()
            self._d2frcnn_path = str(model_path)
            self._d2frcnn_device = dev
            self._d2frcnn_imgsz = imgsz

        return self._predict_with_model(self._d2frcnn_model, image_path, self._d2frcnn_device, self._d2frcnn_imgsz, conf)
