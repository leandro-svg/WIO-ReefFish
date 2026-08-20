# `fish_monitoring` — Core Package

Training, inference, evaluation and visualisation logic for the WIO-ReefFish
detector benchmark.

## Module Map

### Top level

| Module | Purpose |
|--------|---------|
| `cli.py` | Argparse CLI (`python -m fish_monitoring`). Wires every sub-command: train, eval, infer, diagnose, video, benchmark. |
| `constants.py` | Fallback `CLASS_NAMES` (24 fish families) and `DEFAULT_SPLITS`. |

### `core/` — Data types and dataset helpers

| Module | Purpose |
|--------|---------|
| `inference.py` | `Pred` dataclass, NMS, `predict_image` / `predict_image_tiled` for any Ultralytics-style model. |
| `dataset.py` | Image/label counts per split and per-class instance distributions. |
| `labels.py` | YOLO label filtering by box area, with resolution rescaling (`AreaResize`). |

### `eval/` — Evaluation engine

| Module | Purpose |
|--------|---------|
| `kitti_eval.py` | **Core evaluator.** KITTI 11-point interpolated AP with difficulty stratification by GT bbox-height percentiles. Key types: `BoxSet`, `DifficultyResult`. |
| `metrics.py` | YOLO-style metrics: difficulty thresholds, PR/F1 curves, confusion matrices. |
| `diagnose.py` | Confidence sweeping — matches predictions to GT and picks the best threshold by F1 or mAP. |
| `report.py` | Markdown report generation from evaluation results. |
| `spatial.py` | Spatial FP/FN heatmaps — whether errors concentrate in specific image regions. |

### `training/` — Training and inference pipelines

| Module | Purpose |
|--------|---------|
| `yolo_ops.py` | Ultralytics train/validate wrappers, with optional underwater augmentation. |
| `inference.py` | End-to-end inference: auto-selects a confidence threshold, runs a detector on a split, writes YOLO-format predictions. |
| `video.py` | Renders annotated videos from image sequences. |
| `visibility.py` | Crop-based visibility classifier (good / medium / poor). |

### `baselines/` — Detector implementations

All baselines inherit from `BaseDetector` (ABC) and are resolved through the
lazy registry in `registry.py`.

| Baseline | File | Notes |
|----------|------|-------|
| YOLOv8 | `yolov8_baseline.py` | Ultralytics YOLOv8m |
| YOLO11 | `yolo11_baseline.py` | Ultralytics YOLO11m |
| YOLO26 | `yolo26_baseline.py` | Ultralytics YOLO26m |
| YOLO-World | `yolo_world_baseline.py` | Open-vocabulary YOLO-World |
| YOLO-NAS | `yolo_nas_baseline.py` | Super-Gradients YOLO-NAS |
| RT-DETR | `rtdetr_baseline.py` | Ultralytics RT-DETR-l |
| Faster R-CNN | `faster_rcnn_baseline.py` | Torchvision, ResNet-50-FPN |
| RetinaNet | `retinanet_baseline.py` | Torchvision, ResNet-50-FPN |
| DINOv2 | `dinov2_backbone.py` | Frozen DINOv2 ViT + detection head |
| DINOv2 + FRCNN | `dinov2_frcnn.py` | DINOv2 features to Faster R-CNN |
| SAM 2 | `sam2_baseline.py` | Automatic mask to bounding box |
| Grounding DINO | `grounding_dino_baseline.py` | Zero-shot, text-prompted |
| Co-DETR | `co_detr_baseline.py` | Collaborative DETR, trained from scratch |

### `underwater/` — Domain-specific modules (experimental)

| Module | Purpose |
|--------|---------|
| `fwnwd_loss.py` | Frequency-Weighted Non-local Water Degradation loss |
| `udino_modules.py` | U-DINO underwater pre-processing modules |
| `ultralytics_integration.py` | Hooks to inject FWNWD / U-DINO into Ultralytics training |
| `demo.py` | Small demo of the underwater augmentation |

## Adding a New Baseline

1. Create `baselines/<name>_baseline.py` inheriting from `BaseDetector`.
2. Implement `train()`, `evaluate()` and `predict()`.
3. Register it in `baselines/registry.py`:

```python
_LAZY_REGISTRY["my-detector"] = ("fish_monitoring.baselines.my_detector", "MyDetector")
```

The CLI and `scripts/evaluate.py` pick it up automatically.
