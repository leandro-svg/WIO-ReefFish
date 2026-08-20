"""Abstract base class for all detection baselines.

Every baseline must subclass ``BaseDetector`` and implement:
  - ``train()``   – fine-tune / train on the fish dataset
  - ``evaluate()`` – run evaluation on a split, return metrics dict
  - ``predict()``  – single-image inference returning ``Pred``

Additionally provides shared YOLO-format dataset helpers so that all
baselines can consume the same dataset directory structure.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fish_monitoring.constants import CLASS_NAMES
from fish_monitoring.core.inference import Pred


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

@dataclass
class BaselineTrainConfig:
    """Common training parameters for all baselines."""

    data_yaml: Path
    weights: str = ""
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: Any = 0
    patience: int = 20
    project: str = "results"
    name: str = "baseline_run"
    lr: float = 1e-3
    num_classes: int = len(CLASS_NAMES)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    resume: str | None = None


@dataclass
class BaselineEvalConfig:
    """Common evaluation parameters for all baselines."""

    model_path: Path
    data_yaml: Path
    split: str = "test"
    imgsz: int = 640
    device: Any = 0
    conf: float = 0.25
    iou: float = 0.5
    project: str = "results"
    name: str = "baseline_eval"
    num_classes: int = len(CLASS_NAMES)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))


@dataclass
class BaselineInferConfig:
    """Common inference parameters for all baselines."""

    model_path: Path
    source: Path
    output_dir: Path
    imgsz: int = 640
    device: Any = 0
    conf: float = 0.25
    iou: float = 0.5
    save_txt: bool = True
    save_images: bool = False
    num_classes: int = len(CLASS_NAMES)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))


# ---------------------------------------------------------------------------
# YOLO-format dataset helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_data_yaml(yaml_path: Path) -> dict[str, Any]:
    """Parse a YOLO ``data.yaml`` without requiring PyYAML (best-effort).

    Returns keys: path, train, val, test, nc, names.
    """
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    cfg: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key == "nc":
            try:
                cfg["nc"] = int(float(val))
            except ValueError:
                pass
        elif key == "names":
            # Try to parse inline list: [a, b, c]
            if val.startswith("["):
                val = val.strip("[]")
                cfg["names"] = [x.strip().strip("'\"") for x in val.split(",") if x.strip()]
        elif key in ("path", "train", "val", "test"):
            cfg[key] = val.strip("'\"")
    return cfg


def resolve_split_dirs(
    data_yaml: Path,
    split: str,
) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a given split.

    Handles YOLO directory conventions:
      <dataset_root>/<split>/images/
      <dataset_root>/<split>/labels/
    """
    dataset_dir = data_yaml.parent
    cfg = parse_data_yaml(data_yaml)

    # 'path' key can override root
    root = Path(cfg.get("path", str(dataset_dir)))
    if not root.is_absolute():
        root = dataset_dir / root

    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return images_dir, labels_dir


def iter_split_images(data_yaml: Path, split: str) -> list[Path]:
    """List image files in a split."""
    images_dir, _ = resolve_split_dirs(data_yaml, split)
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def read_yolo_labels(label_path: Path, *, im_w: int, im_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read YOLO label file. Returns (xyxy, cls) in absolute pixels."""
    boxes: list[list[float]] = []
    classes: list[int] = []

    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue

        cx = x * im_w
        cy = y * im_h
        bw = w * im_w
        bh = h * im_h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes.append([x1, y1, x2, y2])
        classes.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


def save_yolo_predictions(
    pred: Pred,
    image_path: Path,
    out_labels_dir: Path,
    im_w: int,
    im_h: int,
) -> Path:
    """Save predictions in YOLO txt format (cls x y w h conf) normalized."""
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for (x1, y1, x2, y2), c, cls_id in zip(pred.xyxy, pred.conf, pred.cls):
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        x_n = cx / im_w
        y_n = cy / im_h
        w_n = bw / im_w
        h_n = bh / im_h
        lines.append(f"{int(cls_id)} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f} {float(c):.6f}\n")

    out_path = out_labels_dir / f"{image_path.stem}.txt"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Abstract base detector
# ---------------------------------------------------------------------------

class BaseDetector(abc.ABC):
    """Abstract interface that every baseline detector must implement."""

    name: str = "base"

    @abc.abstractmethod
    def train(self, cfg: BaselineTrainConfig) -> Path:
        """Train the model. Return path to best weights."""
        ...

    @abc.abstractmethod
    def evaluate(self, cfg: BaselineEvalConfig) -> dict[str, float]:
        """Evaluate on a split. Return dict with keys like mAP50, precision, recall."""
        ...

    @abc.abstractmethod
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
        """Single-image prediction returning Pred (xyxy, conf, cls)."""
        ...

    def infer_directory(self, cfg: BaselineInferConfig) -> Path:
        """Run inference on a directory of images. Default implementation
        calls ``self.predict()`` per image and saves YOLO txt files."""
        from PIL import Image

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_labels = cfg.output_dir / "labels"

        images = sorted([p for p in cfg.source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        print(f"[{self.name}] Inferring {len(images)} images -> {cfg.output_dir}")

        for img_path in images:
            pred = self.predict(
                img_path,
                model_path=cfg.model_path,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=cfg.iou,
                device=cfg.device,
            )
            w, h = Image.open(img_path).size
            if cfg.save_txt:
                save_yolo_predictions(pred, img_path, out_labels, im_w=w, im_h=h)

        print(f"[{self.name}] Inference complete. {len(images)} images processed.")
        return cfg.output_dir
