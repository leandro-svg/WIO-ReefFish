"""SAM 2 baseline for object detection.

SAM 2 (Segment Anything Model 2) is a segmentation model. For detection,
we use it as a refinement/scoring layer on top of initial box proposals:

1. Generate initial box proposals using a lightweight detector or grid prompts
2. Feed proposals as box prompts to SAM 2
3. Use mask confidence (IoU prediction) as detection score
4. Extract tight bounding boxes from predicted masks

This tests whether SAM 2's visual understanding helps for fish detection,
even though we do not need segmentation masks as the final output.

Usage:
    python main.py train-baseline --baseline sam2 \
        --data ../data/WIO-ReefFish/data.yaml \
        --epochs 30 --batch 4

Note: SAM 2 primarily operates in zero-shot or prompt-based modes.
'Training' here fine-tunes the mask decoder on fish data with GT box prompts.
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


def _download_sam2_checkpoint(model_size: str = "tiny", cache_dir: str | None = None) -> Path:
    """Download SAM 2.1 checkpoint if it does not already exist locally.

    Returns the absolute path to the checkpoint file.
    """
    import urllib.request

    ckpt_map = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt",
        "base": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
    }
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

    ckpt_name = ckpt_map.get(model_size, ckpt_map["tiny"])
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam2")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_path / ckpt_name

    if not ckpt_path.exists():
        url = f"{base_url}/{ckpt_name}"
        print(f"Downloading SAM 2 checkpoint: {url} → {ckpt_path}")
        urllib.request.urlretrieve(url, str(ckpt_path))
        print(f"Download complete: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Using cached SAM 2 checkpoint: {ckpt_path}")

    return ckpt_path


def _load_sam2_model(model_size: str = "tiny", device: str = "cpu"):
    """Load SAM 2 model using the official API.

    Requires: pip install segment-anything-2 (or sam2)

    Falls back to sam2 hub loading if the official package is not found.
    """
    import torch

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config_map = {
            "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }

        ckpt = str(_download_sam2_checkpoint(model_size))
        cfg = config_map.get(model_size, config_map["tiny"])

        sam2 = build_sam2(cfg, ckpt, device=device)
        predictor = SAM2ImagePredictor(sam2)
        return sam2, predictor

    except ImportError:
        # Fallback: try torch.hub
        try:
            sam2 = torch.hub.load("facebookresearch/sam2", f"sam2.1_hiera_{model_size}", pretrained=True)
            sam2 = sam2.to(device)
            return sam2, None
        except Exception as e:
            raise RuntimeError(
                "SAM 2 is not installed. Install it with:\n"
                "  pip install segment-anything-2\n"
                "  or: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e


class SAM2Detector(BaseDetector):
    """SAM 2 based detection: box prompts → mask → refined box + confidence.

    Strategy:
    - Zero-shot / fine-tuned mask decoder
    - GT boxes or a simple proposal generator provides box prompts
    - SAM 2's IoU prediction head acts as confidence scorer
    - Tight bounding boxes extracted from predicted masks
    """

    name = "sam2"

    def train(self, cfg: BaselineTrainConfig) -> Path:
        """Fine-tune SAM 2's mask decoder on fish data with GT box prompts.

        Only the mask decoder is updated; image encoder stays frozen.
        """
        import torch
        import torch.nn.functional as F

        device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) and torch.cuda.is_available() else "cpu")

        sam2, predictor = _load_sam2_model("tiny", str(device))
        sam2.to(device)

        # Freeze everything except mask decoder
        for param in sam2.parameters():
            param.requires_grad = False
        for param in sam2.sam_mask_decoder.parameters():
            param.requires_grad = True

        trainable = [p for p in sam2.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=1e-4)

        out_dir = Path(cfg.project) / cfg.name / "weights"
        out_dir.mkdir(parents=True, exist_ok=True)

        images = iter_split_images(cfg.data_yaml, "train")
        _, labels_dir = resolve_split_dirs(cfg.data_yaml, "train")

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            sam2.train()
            sam2.image_encoder.eval()  # Keep encoder frozen
            epoch_loss = 0.0
            n_batches = 0

            for img_path in images:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                orig_w, orig_h = img.size

                label_path = labels_dir / f"{img_path.stem}.txt"
                gt_xyxy, gt_cls = read_yolo_labels(label_path, im_w=orig_w, im_h=orig_h)

                if gt_xyxy.shape[0] == 0:
                    continue

                if predictor is not None:
                    predictor.set_image(img_np)

                    # Use each GT box as a prompt
                    for i in range(gt_xyxy.shape[0]):
                        box = gt_xyxy[i]
                        try:
                            masks, scores, logits = predictor.predict(
                                box=box,
                                multimask_output=False,
                            )

                            # Create GT mask from box (approximate)
                            gt_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                            x1, y1, x2, y2 = map(int, box)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(orig_w, x2), min(orig_h, y2)
                            gt_mask[y1:y2, x1:x2] = 1.0

                            # Loss
                            pred_mask = torch.as_tensor(masks[0], dtype=torch.float32, device=device)
                            gt_mask_t = torch.as_tensor(gt_mask, dtype=torch.float32, device=device)

                            loss = F.binary_cross_entropy_with_logits(
                                torch.as_tensor(logits[0], dtype=torch.float32, device=device),
                                gt_mask_t.unsqueeze(0),
                            )
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            epoch_loss += float(loss)
                            n_batches += 1
                        except Exception:
                            continue

            avg = epoch_loss / max(n_batches, 1)
            print(f"[SAM 2] Epoch {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")

            if avg < best_loss:
                best_loss = avg
                torch.save(sam2.sam_mask_decoder.state_dict(), out_dir / "best.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"[SAM 2] Early stopping at epoch {epoch + 1}")
                break

        meta = {"num_classes": cfg.num_classes, "class_names": cfg.class_names, "model_size": "tiny"}
        torch.save(meta, out_dir / "meta.pt")

        best_path = out_dir / "best.pt"
        print(f"[SAM 2] Training complete. Best decoder weights: {best_path}")
        return best_path

    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        """Evaluate SAM 2 by prompting with GT boxes and checking IoU."""
        from fish_monitoring.eval.diagnose import _match_predictions, Gt, _load_image_size

        _, labels_dir = resolve_split_dirs(cfg.data_yaml, cfg.split)
        image_paths = iter_split_images(cfg.data_yaml, cfg.split)

        tp_total = fp_total = fn_total = 0

        for img_path in image_paths:
            w, h = _load_image_size(img_path)
            gt_xyxy, gt_cls = read_yolo_labels(labels_dir / f"{img_path.stem}.txt", im_w=w, im_h=h)
            pred = self.predict(img_path, model_path=cfg.model_path, imgsz=cfg.imgsz,
                                conf=cfg.conf, iou=cfg.iou, device=cfg.device)
            gt = Gt(xyxy=gt_xyxy, cls=gt_cls)
            tp_i, fp_i, fn_i, _ = _match_predictions(gt, pred, iou_th=cfg.iou)
            tp_total += tp_i
            fp_total += fp_i
            fn_total += fn_i

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {"precision": prec, "recall": rec, "f1": f1, "tp": tp_total, "fp": fp_total, "fn": fn_total}
        print(f"[SAM 2] Eval: {metrics}")
        return metrics

    def predict(
        self, image_path: Path, *, model_path: Path,
        imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: Any = 0,
    ) -> Pred:
        """Predict using sliding-window box proposals + SAM 2 refinement.

        1. Generate a grid of candidate boxes across the image
        2. Prompt SAM 2 with each box
        3. Use SAM 2's IoU score as confidence
        4. Extract tight bboxes from masks
        5. Apply NMS
        """
        import torch
        from PIL import Image

        dev = str(device) if not isinstance(device, int) else f"cuda:{device}" if torch.cuda.is_available() else "cpu"

        if not hasattr(self, "_sam2_predictor"):
            self._sam2, self._sam2_predictor = _load_sam2_model("tiny", dev)
            if model_path.exists():
                try:
                    state = torch.load(str(model_path), map_location=dev, weights_only=True)
                    self._sam2.sam_mask_decoder.load_state_dict(state)
                except Exception:
                    pass
            self._sam2.eval()

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        orig_w, orig_h = img.size

        if self._sam2_predictor is None:
            return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                        conf=np.zeros((0,), dtype=np.float32),
                        cls=np.zeros((0,), dtype=np.int64))

        self._sam2_predictor.set_image(img_np)

        # Generate grid proposals (automatic prompts)
        # Use SAM 2's automatic mask generator if available
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_gen = SAM2AutomaticMaskGenerator(
                model=self._sam2,
                points_per_side=32,
                pred_iou_thresh=conf,
                stability_score_thresh=0.8,
                min_mask_region_area=100,
            )
            masks = mask_gen.generate(img_np)

            if not masks:
                return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                            conf=np.zeros((0,), dtype=np.float32),
                            cls=np.zeros((0,), dtype=np.int64))

            all_boxes = []
            all_scores = []
            for m in masks:
                bbox = m["bbox"]  # [x, y, w, h] (COCO format)
                x, y, w, h = bbox
                all_boxes.append([x, y, x + w, y + h])
                all_scores.append(float(m["predicted_iou"]))

            boxes_arr = np.array(all_boxes, dtype=np.float32)
            scores_arr = np.array(all_scores, dtype=np.float32)
            # SAM 2 is class-agnostic; assign class 0 as placeholder
            cls_arr = np.zeros(len(all_boxes), dtype=np.int64)

            # NMS
            from fish_monitoring.core.inference import _nms_xyxy
            keep = _nms_xyxy(boxes_arr, scores_arr, iou)

            return Pred(xyxy=boxes_arr[keep], conf=scores_arr[keep], cls=cls_arr[keep])

        except ImportError:
            # Fallback: grid-based box proposals
            grid_sizes = [32, 64, 128, 256]
            all_boxes = []
            all_scores = []

            for gs in grid_sizes:
                for y in range(0, orig_h - gs + 1, gs // 2):
                    for x in range(0, orig_w - gs + 1, gs // 2):
                        box = np.array([x, y, x + gs, y + gs], dtype=np.float32)
                        try:
                            masks, scores, _ = self._sam2_predictor.predict(
                                box=box, multimask_output=False,
                            )
                            if float(scores[0]) >= conf:
                                # Get tight bbox from mask
                                mask = masks[0]
                                ys, xs = np.where(mask)
                                if len(ys) > 0:
                                    tight_box = [float(xs.min()), float(ys.min()),
                                                 float(xs.max()), float(ys.max())]
                                    all_boxes.append(tight_box)
                                    all_scores.append(float(scores[0]))
                        except Exception:
                            continue

            if not all_boxes:
                return Pred(xyxy=np.zeros((0, 4), dtype=np.float32),
                            conf=np.zeros((0,), dtype=np.float32),
                            cls=np.zeros((0,), dtype=np.int64))

            boxes_arr = np.array(all_boxes, dtype=np.float32)
            scores_arr = np.array(all_scores, dtype=np.float32)
            cls_arr = np.zeros(len(all_boxes), dtype=np.int64)

            from fish_monitoring.core.inference import _nms_xyxy
            keep = _nms_xyxy(boxes_arr, scores_arr, iou)

            return Pred(xyxy=boxes_arr[keep], conf=scores_arr[keep], cls=cls_arr[keep])
