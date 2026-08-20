"""DINOv2 backbone for object detection.

Uses DINOv2 (facebookresearch) as a frozen feature extractor with a
lightweight detection head (Feature Pyramid + classification/regression).
This enables leveraging powerful self-supervised visual representations.

Usage:
    python main.py train-baseline --baseline dinov2 \
        --data ../data/WIO-ReefFish/data.yaml \
        --epochs 60 --batch 8 --lr 0.001
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


# ---------------------------------------------------------------------------
# DINOv2 Feature Extractor + Simple Detection Head
# ---------------------------------------------------------------------------

def _build_dinov2_detector(num_classes: int, backbone_size: str = "small"):
    """Build a DINOv2-based detector with a simple feature pyramid + head.

    Architecture:
    1. DINOv2 ViT backbone (frozen by default)
    2. Feature projection (patch tokens → spatial feature maps)
    3. Simple FPN
    4. Detection head (class + box regression)
    """
    import torch
    import torch.nn as nn

    class DINOv2DetectionHead(nn.Module):
        """Lightweight detection head on top of DINOv2 patch features.

        Converts DINOv2 patch tokens into spatial feature maps, then applies
        a simple anchor-free detection head (similar to FCOS-style).
        """

        def __init__(self, num_classes: int, backbone_size: str = "small"):
            super().__init__()
            self.num_classes = num_classes

            # DINOv2 backbone sizes
            model_map = {
                "small": ("dinov2_vits14", 384),
                "base": ("dinov2_vitb14", 768),
                "large": ("dinov2_vitl14", 1024),
            }
            model_name, feat_dim = model_map.get(backbone_size, model_map["small"])

            # Load DINOv2 backbone
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
            self.backbone.eval()  # Freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

            self.feat_dim = feat_dim
            self.patch_size = 14  # DINOv2 patch size

            # Project features
            self.proj = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            # Detection head (anchor-free, per-cell predictions)
            self.cls_head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1),
            )

            self.reg_head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 4, 1),  # (dx, dy, dw, dh)
            )

            self.obj_head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, 1),
            )

        def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract DINOv2 patch features and reshape to spatial map."""
            B = x.shape[0]
            H_in, W_in = x.shape[2], x.shape[3]

            # Resize to multiple of patch_size
            H = (H_in // self.patch_size) * self.patch_size
            W = (W_in // self.patch_size) * self.patch_size
            if H != H_in or W != W_in:
                x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

            with torch.no_grad():
                features = self.backbone.forward_features(x)
                patch_tokens = features["x_norm_patchtokens"]  # (B, num_patches, feat_dim)

            h_patches = H // self.patch_size
            w_patches = W // self.patch_size
            # Reshape to spatial: (B, feat_dim, h_patches, w_patches)
            feat_map = patch_tokens.transpose(1, 2).reshape(B, self.feat_dim, h_patches, w_patches)
            return feat_map

        def forward(self, x: torch.Tensor):
            """Forward pass. Returns (cls_logits, reg_preds, obj_logits, feat_h, feat_w)."""
            feat_map = self._extract_features(x)
            proj = self.proj(feat_map)

            cls_out = self.cls_head(proj)       # (B, num_classes, H, W)
            reg_out = self.reg_head(proj)       # (B, 4, H, W)
            obj_out = self.obj_head(proj)       # (B, 1, H, W)

            return cls_out, reg_out, obj_out, feat_map.shape[2], feat_map.shape[3]

    return DINOv2DetectionHead(num_classes, backbone_size)


def _decode_predictions(
    cls_out, reg_out, obj_out,
    feat_h: int, feat_w: int,
    img_h: int, img_w: int,
    conf_th: float,
    patch_size: int = 14,
):
    """Decode anchor-free predictions to (xyxy, conf, cls) in pixel coords."""
    import torch

    B = cls_out.shape[0]
    assert B == 1, "Batch decode only supports B=1"

    cls_prob = torch.sigmoid(cls_out[0])  # (num_classes, H, W)
    obj_prob = torch.sigmoid(obj_out[0, 0])  # (H, W)
    reg = reg_out[0]  # (4, H, W)

    num_classes = cls_prob.shape[0]

    # Create grid centers
    stride_h = img_h / feat_h
    stride_w = img_w / feat_w

    yy, xx = torch.meshgrid(
        torch.arange(feat_h, device=cls_prob.device, dtype=torch.float32),
        torch.arange(feat_w, device=cls_prob.device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (xx + 0.5) * stride_w
    cy = (yy + 0.5) * stride_h

    # Decode boxes: reg predicts (left, top, right, bottom) distances
    dl = torch.exp(reg[0]) * stride_w
    dt = torch.exp(reg[1]) * stride_h
    dr = torch.exp(reg[2]) * stride_w
    db = torch.exp(reg[3]) * stride_h

    x1 = cx - dl
    y1 = cy - dt
    x2 = cx + dr
    y2 = cy + db

    # Flatten
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)
    obj_flat = obj_prob.reshape(-1)

    cls_prob_flat = cls_prob.reshape(num_classes, -1).T  # (H*W, num_classes)

    # Score = objectness × class prob
    scores_all = obj_flat.unsqueeze(1) * cls_prob_flat  # (H*W, num_classes)
    max_scores, max_cls = scores_all.max(dim=1)

    keep = max_scores >= conf_th
    if keep.sum() == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], dim=1)
    confs = max_scores[keep]
    classes = max_cls[keep]

    # Clamp
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_h)

    # NMS per class
    from torchvision.ops import batched_nms
    keep_nms = batched_nms(boxes, confs, classes, iou_threshold=0.5)

    return (
        boxes[keep_nms].cpu().numpy().astype(np.float32),
        confs[keep_nms].cpu().numpy().astype(np.float32),
        classes[keep_nms].cpu().numpy().astype(np.int64),
    )


# ---------------------------------------------------------------------------
# YOLO-format dataset for DINOv2 training
# ---------------------------------------------------------------------------

class _DINOv2DetDataset:
    """Dataset for training DINOv2 detection head (anchor-free targets)."""

    def __init__(self, data_yaml: Path, split: str, imgsz: int = 518):
        # DINOv2 expects multiples of 14; 518 = 37 × 14
        self.images = iter_split_images(data_yaml, split)
        _, self.labels_dir = resolve_split_dirs(data_yaml, split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.imgsz, self.imgsz))
        img_tensor = F.to_tensor(img)

        # Normalize for DINOv2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        xyxy, cls = read_yolo_labels(label_path, im_w=orig_w, im_h=orig_h)

        # Scale to new image size
        if xyxy.shape[0] > 0:
            sx = self.imgsz / orig_w
            sy = self.imgsz / orig_h
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

        target = {
            "boxes": torch.as_tensor(xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(cls, dtype=torch.int64),
        }
        return img_tensor, target


def _collate_fn(batch):
    return tuple(zip(*batch))


class DINOv2Detector(BaseDetector):
    """DINOv2 backbone + anchor-free detection head."""

    name = "dinov2"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        import torch
        from torch.utils.data import DataLoader

        device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) and torch.cuda.is_available() else "cpu")

        # DINOv2 uses 14px patches; best imgsz is multiple of 14
        imgsz = ((cfg.imgsz + 13) // 14) * 14
        if imgsz != cfg.imgsz:
            print(f"[DINOv2] Adjusting imgsz from {cfg.imgsz} to {imgsz} (multiple of 14)")

        model = _build_dinov2_detector(cfg.num_classes, backbone_size="small")
        model.to(device)

        train_ds = _DINOv2DetDataset(cfg.data_yaml, "train", imgsz=imgsz)
        valid_ds = _DINOv2DetDataset(cfg.data_yaml, "valid", imgsz=imgsz)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                                  num_workers=min(4, os.cpu_count() or 1), collate_fn=_collate_fn, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=cfg.batch, shuffle=False,
                                  num_workers=min(4, os.cpu_count() or 1), collate_fn=_collate_fn, pin_memory=True)

        # Only train detection heads (backbone frozen)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        out_dir = Path(cfg.project) / cfg.name / "weights"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            # But keep backbone frozen
            model.backbone.eval()
            epoch_loss = 0.0

            for images_batch, targets_batch in train_loader:
                images = torch.stack(images_batch).to(device)
                cls_out, reg_out, obj_out, fh, fw = model(images)

                loss = _compute_loss(cls_out, reg_out, obj_out, fh, fw,
                                     targets_batch, imgsz, cfg.num_classes, device)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
                optimizer.step()
                epoch_loss += float(loss)

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images_batch, targets_batch in valid_loader:
                    images = torch.stack(images_batch).to(device)
                    cls_out, reg_out, obj_out, fh, fw = model(images)
                    loss = _compute_loss(cls_out, reg_out, obj_out, fh, fw,
                                         targets_batch, imgsz, cfg.num_classes, device)
                    val_loss += float(loss)

            avg_train = epoch_loss / max(len(train_loader), 1)
            avg_val = val_loss / max(len(valid_loader), 1)
            print(f"[DINOv2] Epoch {epoch + 1}/{cfg.epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

            if avg_val < best_loss:
                best_loss = avg_val
                # Only save trainable parts
                state = {k: v for k, v in model.state_dict().items() if "backbone" not in k}
                torch.save(state, out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"[DINOv2] Early stopping at epoch {epoch + 1}")
                break

        meta = {"num_classes": cfg.num_classes, "class_names": cfg.class_names,
                "imgsz": imgsz, "backbone_size": "small"}
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[DINOv2] Training complete. Best weights: {best_path}")
        return best_path

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        import torch
        from fish_monitoring.eval.diagnose import _match_predictions, Gt, _load_image_size

        device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) and torch.cuda.is_available() else "cpu")

        meta_path = cfg.model_path.parent / "meta.pt"
        nc = cfg.num_classes
        imgsz = cfg.imgsz
        backbone_size = "small"
        if meta_path.exists():
            meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
            nc = meta.get("num_classes", nc)
            imgsz = meta.get("imgsz", imgsz)
            backbone_size = meta.get("backbone_size", "small")

        imgsz = ((imgsz + 13) // 14) * 14

        model = _build_dinov2_detector(nc, backbone_size)
        state = torch.load(str(cfg.model_path), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        images_dir, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            pred = self._predict_with_model(model, img_path, device, imgsz, cfg.conf)
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

        metrics = {"precision": prec, "recall": rec, "f1": f1, "tp": tp_total, "fp": fp_total, "fn": fn_total}
        print(f"[DINOv2] Eval: {metrics}")
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
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            cls_out, reg_out, obj_out, fh, fw = model(img_tensor)

        xyxy, confs, classes = _decode_predictions(
            cls_out, reg_out, obj_out, fh, fw,
            imgsz, imgsz, conf, model.patch_size,
        )

        # Scale back to original image
        if xyxy.shape[0] > 0:
            sx = orig_w / imgsz
            sy = orig_h / imgsz
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

        return Pred(xyxy=xyxy, conf=confs, cls=classes)

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

        dev = torch.device(f"cuda:{device}" if isinstance(device, int) and torch.cuda.is_available() else "cpu")

        if not hasattr(self, "_dinov2_model") or self._dinov2_path != str(model_path):
            meta_path = model_path.parent / "meta.pt"
            nc = len(CLASS_NAMES)
            bb_size = "small"
            if meta_path.exists():
                meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
                nc = meta.get("num_classes", len(CLASS_NAMES))
                imgsz = meta.get("imgsz", imgsz)
                bb_size = meta.get("backbone_size", "small")

            imgsz = ((imgsz + 13) // 14) * 14
            self._dinov2_model = _build_dinov2_detector(nc, bb_size)
            state = torch.load(str(model_path), map_location=dev, weights_only=True)
            self._dinov2_model.load_state_dict(state, strict=False)
            self._dinov2_model.to(dev)
            self._dinov2_model.eval()
            self._dinov2_path = str(model_path)
            self._dinov2_device = dev
            self._dinov2_imgsz = imgsz

        return self._predict_with_model(self._dinov2_model, image_path, self._dinov2_device, self._dinov2_imgsz, conf)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _compute_loss(cls_out, reg_out, obj_out, feat_h, feat_w, targets, imgsz, num_classes, device):
    """Anchor-free detection loss: focal cls + GIoU regression + objectness."""
    import torch
    import torch.nn.functional as F_nn

    B = cls_out.shape[0]
    stride_h = imgsz / feat_h
    stride_w = imgsz / feat_w

    total_cls_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    total_obj_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        boxes_gt = targets[b]["boxes"].to(device)  # (N, 4) xyxy
        labels_gt = targets[b]["labels"].to(device)  # (N,)

        # Create objectness and class targets
        obj_target = torch.zeros(feat_h, feat_w, device=device)
        cls_target = torch.zeros(num_classes, feat_h, feat_w, device=device)

        if boxes_gt.shape[0] > 0:
            # Assign each GT box to grid cell of its center
            cx_gt = (boxes_gt[:, 0] + boxes_gt[:, 2]) / 2
            cy_gt = (boxes_gt[:, 1] + boxes_gt[:, 3]) / 2
            gx = (cx_gt / stride_w).long().clamp(0, feat_w - 1)
            gy = (cy_gt / stride_h).long().clamp(0, feat_h - 1)

            for i in range(boxes_gt.shape[0]):
                obj_target[gy[i], gx[i]] = 1.0
                cls_target[labels_gt[i], gy[i], gx[i]] = 1.0

        # Objectness loss (BCE)
        obj_pred = obj_out[b, 0]  # (H, W)
        total_obj_loss += F_nn.binary_cross_entropy_with_logits(obj_pred, obj_target)

        # Classification loss (focal-like BCE on positive cells)
        cls_pred = cls_out[b]  # (C, H, W)
        # Use focal loss weight
        p = torch.sigmoid(cls_pred)
        ce = F_nn.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none")
        focal_weight = (1 - p) ** 2 * cls_target + p ** 2 * (1 - cls_target)
        total_cls_loss += (focal_weight * ce).mean()

        # Regression loss on positive cells only
        if boxes_gt.shape[0] > 0:
            reg_pred = reg_out[b]  # (4, H, W)
            reg_losses = []
            for i in range(boxes_gt.shape[0]):
                gi, gj = int(gx[i]), int(gy[i])

                # Decode predicted box
                pred_dl = torch.exp(reg_pred[0, gj, gi]) * stride_w
                pred_dt = torch.exp(reg_pred[1, gj, gi]) * stride_h
                pred_dr = torch.exp(reg_pred[2, gj, gi]) * stride_w
                pred_db = torch.exp(reg_pred[3, gj, gi]) * stride_h

                cell_cx = (gi + 0.5) * stride_w
                cell_cy = (gj + 0.5) * stride_h

                pred_x1 = cell_cx - pred_dl
                pred_y1 = cell_cy - pred_dt
                pred_x2 = cell_cx + pred_dr
                pred_y2 = cell_cy + pred_db

                # GIoU loss
                gt_box = boxes_gt[i]
                inter_x1 = torch.max(pred_x1, gt_box[0])
                inter_y1 = torch.max(pred_y1, gt_box[1])
                inter_x2 = torch.min(pred_x2, gt_box[2])
                inter_y2 = torch.min(pred_y2, gt_box[3])

                inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
                pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
                gt_area = (gt_box[2] - gt_box[0]).clamp(min=0) * (gt_box[3] - gt_box[1]).clamp(min=0)
                union = pred_area + gt_area - inter_area + 1e-7
                iou = inter_area / union

                # Enclosing box
                enc_x1 = torch.min(pred_x1, gt_box[0])
                enc_y1 = torch.min(pred_y1, gt_box[1])
                enc_x2 = torch.max(pred_x2, gt_box[2])
                enc_y2 = torch.max(pred_y2, gt_box[3])
                enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-7

                giou = iou - (enc_area - union) / enc_area
                reg_losses.append(1.0 - giou)

            if reg_losses:
                total_reg_loss += torch.stack(reg_losses).mean()

    total = (total_cls_loss + total_reg_loss * 5.0 + total_obj_loss) / max(B, 1)
    return total
