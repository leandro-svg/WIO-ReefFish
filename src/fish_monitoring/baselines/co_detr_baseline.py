"""Co-DETR baseline — Collaborative DEtection TRansformer.

Co-DETR (Zong et al., ICCV 2023) enhances DETR by learning
collaborative hybrid assignments from multiple auxiliary heads
(one-to-many assignment + one-to-one set matching).  This makes
the transformer encoder more discriminative and converges faster.

Because Co-DETR requires mmdetection and mmcv (complex compiled deps),
we implement a **practical proxy** that captures the key architectural
idea:

    ResNet-50 FPN backbone  →  Deformable DETR head  +  auxiliary
    one-to-many RetinaNet head (collaborative training only)

At inference time only the DETR head is used; the auxiliary head is
discarded.  This preserves the *collaborative* spirit while staying
dependency-light (just torchvision + torch).

Usage:
    python main.py train-baseline --baseline co-detr \
        --data data/WIO-ReefFish/data.yaml --epochs 50 --batch 4
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from fish_monitoring.constants import CLASS_NAMES

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
from fish_monitoring.core.inference import Pred

# Re-use dataset adapter
from fish_monitoring.baselines.faster_rcnn_baseline import (
    _YOLODetectionDataset,
    _collate_fn,
)


# ---------------------------------------------------------------------------
# Lightweight Deformable-DETR–style detection head
# ---------------------------------------------------------------------------


def _build_co_detr_model(num_classes: int, num_queries: int = 100):
    """Build a simplified Co-DETR model.

    Architecture:
    - ResNet-50 + FPN backbone (torchvision)
    - Transformer decoder with learned object queries
    - MLP heads for classification + bbox regression
    - Auxiliary one-to-many (dense) head for collaborative training

    Key design choices (matching DETR best-practices):
    - Sigmoid focal-loss classification (no background class)
    - L1 + GIoU box loss
    - Proper Hungarian matching via scipy
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.ops import FeaturePyramidNetwork

    class _FPNBackbone(nn.Module):
        """ResNet-50 + FPN backbone producing multi-scale features."""

        def __init__(self):
            super().__init__()
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Extract layers
            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            )
            self.layer1 = resnet.layer1  # 256-ch
            self.layer2 = resnet.layer2  # 512-ch
            self.layer3 = resnet.layer3  # 1024-ch
            self.layer4 = resnet.layer4  # 2048-ch

            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[256, 512, 1024, 2048],
                out_channels=256,
            )

        def forward(self, x):
            c1 = self.layer0(x)
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)

            fpn_feats = self.fpn({
                "feat2": c2,
                "feat3": c3,
                "feat4": c4,
                "feat5": c5,
            })
            return fpn_feats

    class _TransformerDecoder(nn.Module):
        """Simple 6-layer transformer decoder operating on flattened FPN features.

        Supports gradient checkpointing to trade compute for memory.
        """

        def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self.use_checkpoint = True  # gradient checkpointing

        def forward(self, memory, queries):
            # memory: (B, S, D), queries: (B, Q, D)
            from torch.utils.checkpoint import checkpoint
            out = queries
            for layer in self.layers:
                if self.use_checkpoint and self.training:
                    out = checkpoint(layer, out, memory, use_reentrant=False)
                else:
                    out = layer(out, memory)
            return out

    class CoDETR(nn.Module):
        def __init__(self, nc: int, nq: int):
            super().__init__()
            self.backbone = _FPNBackbone()
            self.decoder = _TransformerDecoder(d_model=256)
            self.query_embed = nn.Embedding(nq, 256)
            self.num_queries = nq
            self.num_classes = nc

            # Detection heads — sigmoid per-class (NO background class)
            self.class_head = nn.Linear(256, nc)
            self.bbox_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4),  # (cx, cy, w, h) normalised
            )

            # Auxiliary dense head (collaborative one-to-many)
            self.aux_cls = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, nc, 1),
            )
            self.aux_box = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 4, 1),
            )

            # Positional encoding
            self.pos_enc = nn.Parameter(torch.randn(1, 40000, 256) * 0.02)

            # Bias init: make initial sigmoid output ≈ 0.01 (prior_prob trick)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.class_head.bias, bias_value)
            nn.init.constant_(self.aux_cls[-1].bias, bias_value)

        def forward(self, images, targets=None):
            B = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

            if isinstance(images, list):
                images = torch.stack(images)

            fpn_feats = self.backbone(images)

            # Flatten FPN features → (B, S, 256)
            flat_feats = []
            feat_list = list(fpn_feats.values())
            for feat in feat_list:
                B_, C_, H_, W_ = feat.shape
                flat_feats.append(feat.flatten(2).permute(0, 2, 1))  # (B, H*W, C)

            memory = torch.cat(flat_feats, dim=1)  # (B, S_total, 256)

            # Add positional encoding
            S = memory.shape[1]
            if S <= self.pos_enc.shape[1]:
                memory = memory + self.pos_enc[:, :S, :]
            else:
                # Interpolate positional encoding
                pe = self.pos_enc.permute(0, 2, 1)
                pe = nn.functional.interpolate(pe, size=S, mode="linear", align_corners=False)
                memory = memory + pe.permute(0, 2, 1)

            # Object queries
            queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

            # Decoder
            hs = self.decoder(memory, queries)  # (B, Q, 256)

            # Heads
            pred_logits = self.class_head(hs)  # (B, Q, nc+1)
            pred_boxes = self.bbox_head(hs).sigmoid()  # (B, Q, 4) normalised

            out = {
                "pred_logits": pred_logits,
                "pred_boxes": pred_boxes,
            }

            if targets is not None:
                # Compute losses
                losses = self._compute_losses(pred_logits, pred_boxes, targets, images.shape[-2:])

                # Auxiliary dense head losses (collaborative training)
                aux_loss = self._compute_aux_losses(feat_list, targets, images.shape[-2:])
                losses["loss_aux"] = aux_loss

                return losses

            return out

        def _compute_losses(self, pred_logits, pred_boxes, targets, img_size):
            """Hungarian matching + sigmoid focal loss + L1 + GIoU loss."""
            import torch
            import torch.nn.functional as F
            from scipy.optimize import linear_sum_assignment
            from torchvision.ops import generalized_box_iou

            device = pred_logits.device
            B = pred_logits.shape[0]
            total_cls_loss = torch.tensor(0.0, device=device)
            total_box_loss = torch.tensor(0.0, device=device)
            total_giou_loss = torch.tensor(0.0, device=device)
            n_matched = 0

            for b in range(B):
                gt_boxes = targets[b]["boxes"]  # xyxy
                gt_labels = targets[b]["labels"]  # 1-indexed

                n_q = pred_boxes.shape[1]

                if gt_boxes.shape[0] == 0:
                    # Only classification loss → all queries should predict low
                    total_cls_loss = total_cls_loss + self._focal_loss(
                        pred_logits[b],
                        torch.zeros_like(pred_logits[b]),
                    )
                    continue

                # Convert gt xyxy → cxcywh normalised
                H, W = img_size
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / W
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / H
                gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]) / W
                gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]) / H
                gt_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)

                n_gt = gt_cxcywh.shape[0]

                # ── Hungarian matching ──
                with torch.no_grad():
                    # Cast to float32 for numerically stable matching
                    pred_boxes_f = pred_boxes[b].float()
                    pred_logits_f = pred_logits[b].float()

                    # L1 cost
                    cost_bbox = torch.cdist(
                        pred_boxes_f, gt_cxcywh.float(), p=1
                    )  # (Q, G)

                    # Focal-loss-based classification cost
                    out_prob = pred_logits_f.sigmoid().clamp(1e-6, 1 - 1e-6)
                    alpha, gamma = 0.25, 2.0
                    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob).log())
                    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-out_prob.log())
                    # gt_labels are 1-indexed → convert to 0-indexed
                    gt_cls_0 = (gt_labels - 1).long()
                    cost_cls = pos_cost_class[:, gt_cls_0] - neg_cost_class[:, gt_cls_0]  # (Q, G)

                    # GIoU cost
                    pred_xyxy_b = self._cxcywh_to_xyxy(pred_boxes_f)
                    gt_xyxy_norm = self._cxcywh_to_xyxy(gt_cxcywh.float())
                    cost_giou = -generalized_box_iou(pred_xyxy_b, gt_xyxy_norm)  # (Q, G)

                    cost = cost_bbox * 5.0 + cost_cls * 2.0 + cost_giou * 2.0
                    # Sanitise NaN/Inf from AMP fp16→fp32 cast edge cases
                    cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
                    cost_np = cost.cpu().numpy()

                    row_ind, col_ind = linear_sum_assignment(cost_np)

                if len(row_ind) == 0:
                    continue

                mq = torch.tensor(row_ind, device=device, dtype=torch.long)
                mg = torch.tensor(col_ind, device=device, dtype=torch.long)

                # ── Sigmoid focal classification loss ──
                cls_target = torch.zeros_like(pred_logits[b])  # (Q, nc) all zeros
                gt_cls_0 = (gt_labels[mg] - 1).long()
                cls_target[mq, gt_cls_0] = 1.0
                total_cls_loss = total_cls_loss + self._focal_loss(
                    pred_logits[b], cls_target
                )

                # ── Box L1 loss (matched only) ──
                total_box_loss = total_box_loss + F.l1_loss(
                    pred_boxes[b][mq], gt_cxcywh[mg]
                )

                # ── GIoU loss (matched only) ──
                pred_xyxy_matched = self._cxcywh_to_xyxy(pred_boxes[b][mq])
                gt_xyxy_matched = self._cxcywh_to_xyxy(gt_cxcywh[mg])
                giou = generalized_box_iou(pred_xyxy_matched, gt_xyxy_matched)
                total_giou_loss = total_giou_loss + (1 - giou.diag()).mean()

                n_matched += len(mq)

            losses = {
                "loss_cls": total_cls_loss / max(B, 1),
                "loss_bbox": total_box_loss / max(B, 1) * 5.0,
                "loss_giou": total_giou_loss / max(B, 1) * 2.0,
            }
            return losses

        @staticmethod
        def _focal_loss(logits, targets, alpha=0.25, gamma=2.0):
            """Sigmoid focal loss for all queries."""
            import torch
            import torch.nn.functional as F
            p = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * ((1 - p_t) ** gamma) * ce
            return loss.mean()

        @staticmethod
        def _cxcywh_to_xyxy(boxes):
            """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
            import torch
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        def _compute_aux_losses(self, fpn_feats, targets, img_size):
            """Auxiliary dense head loss (focal-loss style) for collaborative training."""
            import torch
            import torch.nn.functional as F

            device = fpn_feats[0].device
            total_aux = torch.tensor(0.0, device=device)

            # Use only the largest FPN level (last one) for efficiency
            feat = fpn_feats[-1]  # (B, 256, H', W')
            aux_cls_out = self.aux_cls(feat)  # (B, nc, H', W')

            B, nc, fH, fW = aux_cls_out.shape
            H, W = img_size

            for b in range(B):
                gt_boxes = targets[b]["boxes"]
                gt_labels = targets[b]["labels"]

                # Create one-hot target map (fH, fW, nc) — all zeros by default
                cls_target = torch.zeros(nc, fH, fW, device=device)

                if gt_boxes.shape[0] > 0:
                    for i in range(gt_boxes.shape[0]):
                        cx = ((gt_boxes[i, 0] + gt_boxes[i, 2]) / 2 / W * fW).long().clamp(0, fW - 1)
                        cy = ((gt_boxes[i, 1] + gt_boxes[i, 3]) / 2 / H * fH).long().clamp(0, fH - 1)
                        cls_idx = (gt_labels[i] - 1).long().clamp(0, nc - 1)
                        cls_target[cls_idx, cy, cx] = 1.0

                # Sigmoid focal loss on the dense map
                total_aux = total_aux + self._focal_loss(
                    aux_cls_out[b], cls_target
                ) * 0.1  # down-weight auxiliary

            return total_aux / max(B, 1)

    return CoDETR(nc=num_classes, nq=num_queries)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CoDETRDetector(BaseDetector):
    """Co-DETR (collaborative DETR) baseline."""

    name = "co-detr"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        import torch
        from torch.utils.data import DataLoader

        device = torch.device(
            f"cuda:{cfg.device}"
            if isinstance(cfg.device, int) and torch.cuda.is_available()
            else "cpu"
        )

        model = _build_co_detr_model(cfg.num_classes)
        model.to(device)

        # NOTE: DataParallel is NOT used because CoDETR.forward()
        # receives `targets` as a list-of-dicts, which DP cannot scatter.
        # AMP (fp16) causes NaN in focal-loss / GIoU after ~40 epochs,
        # so we train in full fp32.  RTX 4090 24 GB can handle batch=4 fp32.
        accum_steps = 2  # effective batch = cfg.batch * accum_steps
        print(f"[Co-DETR] Single-GPU fp32 + grad-checkpoint + "
              f"accum_steps={accum_steps}  (eff. batch={cfg.batch * accum_steps})")

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

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=1e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )

        out_dir = Path(cfg.project) / cfg.name / "weights"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, (images, targets) in enumerate(train_loader):
                images = torch.stack([img.to(device) for img in images])
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets=targets)
                losses = sum(v for v in loss_dict.values()) / accum_steps

                # Guard against NaN — skip this batch instead of corrupting weights
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"  [Co-DETR] NaN/Inf loss at step {step}, skipping batch", flush=True)
                    optimizer.zero_grad()
                    continue

                losses.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += float(losses) * accum_steps

            lr_scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in valid_loader:
                    images = torch.stack([img.to(device) for img in images])
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    model.train()  # need train mode for loss computation
                    loss_dict = model(images, targets=targets)
                    vl = sum(v for v in loss_dict.values())
                    if not (torch.isnan(vl) or torch.isinf(vl)):
                        val_loss += float(vl)
                    model.eval()

            avg_train = epoch_loss / max(len(train_loader), 1)
            avg_val = val_loss / max(len(valid_loader), 1)
            print(
                f"[Co-DETR] Epoch {epoch + 1}/{cfg.epochs}  "
                f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}",
                flush=True,
            )

            # Skip saving if loss is NaN/Inf
            if math.isnan(avg_val) or math.isinf(avg_val):
                patience_counter += 1
            elif avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(model.state_dict(), out_dir / "last.pt")

            if patience_counter >= cfg.patience:
                print(f"[Co-DETR] Early stopping at epoch {epoch + 1}")
                break

        meta = {
            "num_classes": cfg.num_classes,
            "class_names": cfg.class_names,
            "imgsz": cfg.imgsz,
        }
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[Co-DETR] Training complete. Best weights: {best_path}")
        return best_path

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        import torch
        from fish_monitoring.eval.diagnose import Gt, _load_image_size, _match_predictions

        device = torch.device(
            f"cuda:{cfg.device}"
            if isinstance(cfg.device, int) and torch.cuda.is_available()
            else "cpu"
        )

        model = _build_co_detr_model(cfg.num_classes)
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
        print(f"[Co-DETR] Eval: {metrics}")
        return metrics

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
            outputs = model(img_tensor)

        pred_logits = outputs["pred_logits"][0]  # (Q, nc)
        pred_boxes = outputs["pred_boxes"][0]  # (Q, 4) cxcywh normalised

        # Sigmoid → confidence (no background class)
        probs = torch.sigmoid(pred_logits)  # (Q, nc)
        scores, cls_ids = probs.max(dim=-1)  # (Q,), (Q,)

        keep = scores >= conf
        scores = scores[keep].cpu().numpy()
        cls_ids = cls_ids[keep].cpu().numpy()
        boxes = pred_boxes[keep].cpu().numpy()

        if boxes.shape[0] == 0:
            return Pred(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int64),
            )

        # cxcywh normalised → xyxy pixels
        cx = boxes[:, 0] * orig_w
        cy = boxes[:, 1] * orig_h
        bw = boxes[:, 2] * orig_w
        bh = boxes[:, 3] * orig_h
        xyxy = np.stack([
            cx - bw / 2, cy - bh / 2,
            cx + bw / 2, cy + bh / 2,
        ], axis=1).astype(np.float32)

        return Pred(
            xyxy=xyxy,
            conf=scores.astype(np.float32),
            cls=cls_ids.astype(np.int64),
        )

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

        if not hasattr(self, "_model") or self._co_detr_path != str(model_path):
            meta_path = model_path.parent / "meta.pt"
            nc = len(CLASS_NAMES)
            if meta_path.exists():
                meta = torch.load(str(meta_path), map_location="cpu", weights_only=True)
                nc = meta.get("num_classes", len(CLASS_NAMES))

            self._co_detr_model = _build_co_detr_model(nc)
            self._co_detr_model.load_state_dict(
                torch.load(str(model_path), map_location=dev, weights_only=True)
            )
            self._co_detr_model.to(dev)
            self._co_detr_model.eval()
            self._co_detr_path = str(model_path)
            self._co_detr_device = dev

        return self._predict_with_model(
            self._co_detr_model, image_path, self._co_detr_device, imgsz, conf
        )
