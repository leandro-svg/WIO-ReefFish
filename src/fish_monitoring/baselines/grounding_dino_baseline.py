"""Grounding DINO baseline — open-vocabulary detector.

Grounding DINO (Liu et al., 2023) combines a DINO-style transformer
detector with grounded pre-training, enabling text-prompted detection.
We evaluate it in two modes:

1. **Zero-shot** – pass fish family names as text prompts, no fine-tuning.
2. **Fine-tuned** – freeze the text encoder and fine-tune visual components
   with detection losses on the fish dataset.

This implementation uses HuggingFace ``transformers``
(``AutoModelForZeroShotObjectDetection``) which ships Grounding DINO
natively — no extra compiled packages required.

Usage (zero-shot):
    python main.py eval-baseline --baseline grounding-dino \
        --model auto --data data/WIO-ReefFish/data.yaml --split test

Usage (fine-tune):
    python main.py train-baseline --baseline grounding-dino \
        --data data/WIO-ReefFish/data.yaml --epochs 20 --batch 4
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
from fish_monitoring.core.inference import Pred

# ---------------------------------------------------------------------------
# HuggingFace model id
# ---------------------------------------------------------------------------

_HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_text_prompt(class_names: list[str]) -> str:
    """Build a period-separated text prompt for Grounding DINO.

    E.g. "Acanthuridae . Balistidae . Labridae ."
    Grounding DINO uses '.' as a token separator.
    """
    return " . ".join(class_names) + " ."


def _map_gdino_labels(
    predicted_text: list[str], class_names: list[str]
) -> np.ndarray:
    """Map Grounding DINO phrase outputs to integer class ids."""
    name_lower = [n.lower() for n in class_names]
    cls_ids: list[int] = []
    for phrase in predicted_text:
        phrase_l = phrase.lower().strip()
        best_cls = 0
        best_overlap = 0
        for ci, cn in enumerate(name_lower):
            if cn in phrase_l or phrase_l in cn:
                overlap = len(cn)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cls = ci
        cls_ids.append(best_cls)
    return np.array(cls_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class GroundingDINODetector(BaseDetector):
    """Grounding DINO baseline with text-prompted detection."""

    name = "grounding-dino"

    # ── training (fine-tune visual backbone, freeze text encoder) ─────────
    def train(self, cfg: BaselineTrainConfig) -> Path:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        device = torch.device(
            f"cuda:{cfg.device}"
            if isinstance(cfg.device, int) and torch.cuda.is_available()
            else "cpu"
        )

        # Load model + processor from HuggingFace
        print(f"[Grounding DINO] Loading {_HF_MODEL_ID} from HuggingFace ...")
        processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_HF_MODEL_ID)
        model.to(device)

        text_prompt = _build_text_prompt(cfg.class_names)

        # Freeze text encoder (bert) parameters
        for name_p, param in model.named_parameters():
            if "text" in name_p.lower() or "bert" in name_p.lower():
                param.requires_grad = False

        # Multi-GPU support
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[Grounding DINO] Using DataParallel across {n_gpus} GPUs")
            model = torch.nn.DataParallel(model)

        # Mixed-precision scaler
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        # Use the YOLO dataset adapter from Faster R-CNN
        from fish_monitoring.baselines.faster_rcnn_baseline import (
            _YOLODetectionDataset,
            _collate_fn,
        )

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

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)
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

            for images, targets in train_loader:
                images_dev = [img.to(device) for img in images]
                targets_dev = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                optimizer.zero_grad()

                # Compute a surrogate loss (Grounding DINO HF API
                # doesn't expose a native training loss with targets).
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    losses = self._compute_surrogate_loss(
                        model, processor, images_dev, targets_dev,
                        text_prompt, device, cfg.imgsz,
                    )

                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(losses)

            lr_scheduler.step()

            # Validation
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for images, targets in valid_loader:
                    images_dev = [img.to(device) for img in images]
                    targets_dev = [
                        {k: v.to(device) for k, v in t.items()} for t in targets
                    ]
                    model.train()
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        vl = self._compute_surrogate_loss(
                            model, processor, images_dev, targets_dev,
                            text_prompt, device, cfg.imgsz,
                        )
                    val_loss += float(vl)
                    model.eval()

            avg_train = epoch_loss / max(len(train_loader), 1)
            avg_val = val_loss / max(len(valid_loader), 1)
            print(
                f"[Grounding DINO] Epoch {epoch + 1}/{cfg.epochs}  "
                f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
            )

            # Unwrap DataParallel for saving
            raw_model = model.module if hasattr(model, "module") else model

            save_loss = avg_val if avg_val > 0 else avg_train
            if save_loss < best_loss:
                best_loss = save_loss
                torch.save(raw_model.state_dict(), out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(raw_model.state_dict(), out_dir / "last.pt")

            if patience_counter >= cfg.patience:
                print(f"[Grounding DINO] Early stopping at epoch {epoch + 1}")
                break

        meta = {
            "num_classes": cfg.num_classes,
            "class_names": cfg.class_names,
            "imgsz": cfg.imgsz,
            "text_prompt": text_prompt,
        }
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[Grounding DINO] Training complete. Best weights: {best_path}")
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

        # Load model
        model, processor, class_names = self._load_model_for_eval(cfg, device)

        _, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        text_prompt = _build_text_prompt(class_names)
        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            pred = self._predict_single(
                model, processor, img_path, text_prompt, class_names,
                device, cfg.imgsz, cfg.conf,
            )
            w, h = _load_image_size(img_path)

            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=w, im_h=h)

            # Zero-shot mode: class-agnostic matching (all GT → cls 0)
            if getattr(self, "_zero_shot", False):
                gt_cls = np.zeros_like(gt_cls)

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
        print(f"[Grounding DINO] Eval: {metrics}")
        return metrics

    # ── single-image prediction (public API) ─────────────────────────────
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

        if not hasattr(self, "_gdino_model") or self._gdino_path != str(model_path):
            from fish_monitoring.constants import CLASS_NAMES as default_names

            # Zero-shot mode: use generic "fish" prompt
            if model_path.name == "auto":
                class_names = ["fish"]
            else:
                # Try to load metadata
                meta_path = model_path.parent / "meta.pt"
                class_names = list(default_names)
                if meta_path.exists():
                    meta = torch.load(
                        str(meta_path), map_location="cpu", weights_only=True
                    )
                    class_names = meta.get("class_names", class_names)

            self._gdino_class_names = class_names
            self._gdino_text_prompt = _build_text_prompt(class_names)

            cfg_stub = type(
                "C",
                (),
                {
                    "model_path": model_path,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                },
            )()
            self._gdino_model, self._gdino_processor, _ = (
                self._load_model_for_eval(cfg_stub, dev)
            )
            self._gdino_path = str(model_path)
            self._gdino_device = dev

        return self._predict_single(
            self._gdino_model,
            self._gdino_processor,
            image_path,
            self._gdino_text_prompt,
            self._gdino_class_names,
            self._gdino_device,
            imgsz,
            conf,
        )

    # ── internal helpers ─────────────────────────────────────────────────

    def _load_model_for_eval(self, cfg, device):
        """Load Grounding DINO for evaluation / inference via HuggingFace."""
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from fish_monitoring.constants import CLASS_NAMES as default_names

        class_names = getattr(cfg, "class_names", list(default_names))
        model_path = Path(str(cfg.model_path))

        # Zero-shot mode: use generic "fish" prompt (pretrained model)
        self._zero_shot = model_path.name == "auto"
        if self._zero_shot:
            class_names = ["fish"]
            print("[Grounding DINO] Zero-shot mode — using generic 'fish' prompt")

        print(f"[Grounding DINO] Loading {_HF_MODEL_ID} from HuggingFace ...")
        processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_HF_MODEL_ID)

        # Load fine-tuned weights on top if available
        if not self._zero_shot and model_path.exists():
            state = torch.load(
                str(model_path), map_location=device, weights_only=True
            )
            model.load_state_dict(state, strict=False)
            print(f"[Grounding DINO] Loaded fine-tuned weights: {model_path}")

        model.to(device)
        model.eval()
        return model, processor, class_names

    def _predict_single(
        self,
        model,
        processor,
        image_path: Path,
        text_prompt: str,
        class_names: list[str],
        device,
        imgsz: int,
        conf: float,
    ) -> Pred:
        """Run Grounding DINO inference on a single image via HuggingFace."""
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Processor handles resizing & normalisation
        inputs = processor(images=img, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: convert to boxes + scores + labels
        # transformers ≥5.x renamed box_threshold → threshold
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=conf,
            text_threshold=conf,
            target_sizes=[(orig_h, orig_w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # xyxy pixels
        scores = results["scores"].cpu().numpy()
        labels_text = results["labels"]  # list of strings

        if boxes.shape[0] == 0:
            return Pred(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int64),
            )

        cls_ids = _map_gdino_labels(labels_text, class_names)

        return Pred(
            xyxy=boxes.astype(np.float32),
            conf=scores.astype(np.float32),
            cls=cls_ids,
        )

    def _compute_surrogate_loss(
        self, model, processor, images, targets, text_prompt, device, imgsz
    ):
        """Compute a surrogate detection loss for fine-tuning.

        The HuggingFace Grounding DINO API doesn't expose a native
        training loss with box targets.  We forward each image through
        the model, extract predicted boxes & logits, then compute L1 +
        classification losses against GT via greedy Hungarian matching.
        """
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from scipy.optimize import linear_sum_assignment

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n = 0

        # Unwrap DataParallel
        raw_model = model.module if hasattr(model, "module") else model

        for img_t, tgt in zip(images, targets):
            gt_boxes = tgt["boxes"]  # xyxy pixels at imgsz scale
            gt_labels = tgt["labels"]
            if gt_boxes.shape[0] == 0:
                continue

            # Convert tensor to PIL for processor
            pil_img = TF.to_pil_image(img_t.cpu())
            inputs = processor(
                images=pil_img, text=text_prompt, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = raw_model(**inputs)

            # outputs.pred_boxes: (1, Q, 4) in cxcywh normalised
            # outputs.logits: (1, Q, num_tokens)
            pred_boxes = outputs.pred_boxes[0]  # (Q, 4)
            pred_logits = outputs.logits[0]  # (Q, T)

            # Convert gt xyxy → cxcywh normalised
            gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / imgsz
            gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / imgsz
            gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]) / imgsz
            gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]) / imgsz
            gt_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)

            # L1 cost matrix
            n_gt = gt_cxcywh.shape[0]
            cost = torch.cdist(
                pred_boxes[:, :4].float(), gt_cxcywh.float(), p=1
            )  # (Q, G)

            # Hungarian matching (scipy – optimal)
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            mq = row_ind.tolist()
            mg = col_ind.tolist()

            if not mq:
                continue

            mq_t = torch.tensor(mq, device=device)
            mg_t = torch.tensor(mg, device=device)

            # Box L1 loss
            box_loss = F.l1_loss(pred_boxes[mq_t, :4], gt_cxcywh[mg_t])

            # Confidence loss on MATCHED queries → push toward 1
            max_logits = pred_logits.max(dim=-1).values  # (Q,)
            matched_logits = max_logits[mq_t]
            target_ones = torch.ones_like(matched_logits)
            pos_loss = F.binary_cross_entropy_with_logits(
                matched_logits, target_ones
            )

            # Light negative loss on UNMATCHED queries → push toward 0
            # (down-weighted 0.1× to avoid old bug of killing all logits)
            all_idx = set(range(pred_logits.shape[0]))
            unmatched_idx = sorted(all_idx - set(mq))
            if unmatched_idx:
                um_t = torch.tensor(unmatched_idx, device=device)
                unmatched_logits = max_logits[um_t]
                neg_loss = F.binary_cross_entropy_with_logits(
                    unmatched_logits, torch.zeros_like(unmatched_logits)
                )
            else:
                neg_loss = torch.tensor(0.0, device=device)

            conf_loss = pos_loss + 0.1 * neg_loss

            total_loss = total_loss + box_loss * 5.0 + conf_loss
            n += 1

        return total_loss / max(n, 1)
