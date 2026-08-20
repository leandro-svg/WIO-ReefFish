from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for fish_monitoring.underwater") from e


@dataclass(frozen=True)
class FWNWDConfig:
    """Configuration for the FWNWD loss.

    Notes:
    - Boxes are expected in xyxy pixel coordinates unless specified otherwise.
    - This implementation is intended for experimentation and ablation.
    """

    box_format: Literal["xyxy", "xywh"] = "xyxy"
    reduction: Literal["mean", "sum", "none"] = "mean"
    eps: float = 1e-7

    # Focaler-IoU style weighting (hard vs easy)
    focal_gamma: float = 2.0

    # WIoU-style distance weighting
    wiou_distance_weight: float = 1.0

    # NWD term weight
    nwd_lambda: float = 1.0

    # Normalization used inside NWD (bigger -> smaller penalty)
    nwd_normalizer: float = 200.0


def _to_xyxy(boxes, *, fmt: str, eps: float):
    torch = _try_import_torch()
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    if fmt == "xyxy":
        return boxes

    if fmt == "xywh":
        x, y, w, h = boxes.unbind(-1)
        w = w.clamp_min(eps)
        h = h.clamp_min(eps)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    raise ValueError(f"Unsupported box format: {fmt}")


def _box_iou_xyxy(a, b, *, eps: float):
    """Pairwise IoU for aligned boxes: a and b are (N,4) in xyxy."""
    torch = _try_import_torch()
    if a.shape != b.shape:
        raise ValueError(f"Expected aligned boxes with same shape, got {a.shape} vs {b.shape}")

    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])

    inter_w = (x2 - x1).clamp_min(0)
    inter_h = (y2 - y1).clamp_min(0)
    inter = inter_w * inter_h

    a_area = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    b_area = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)

    union = (a_area + b_area - inter).clamp_min(eps)
    return inter / union


def _center_wh_xyxy(xyxy, *, eps: float):
    torch = _try_import_torch()
    x1, y1, x2, y2 = xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


def nwd_loss_aligned(pred_boxes, tgt_boxes, *, fmt: str = "xyxy", eps: float = 1e-7, normalizer: float = 200.0):
    """Normalized Wasserstein Distance loss for aligned boxes.

    Uses a common approximation (box as axis-aligned Gaussian):
      W2^2 = (cx_p-cx_t)^2 + (cy_p-cy_t)^2 + (w_p-w_t)^2/4 + (h_p-h_t)^2/4
    Then a similarity is computed as exp(-sqrt(W2^2)/normalizer) and loss = 1 - sim.

    Returns per-box loss with shape (N,).
    """

    torch = _try_import_torch()

    p = _to_xyxy(pred_boxes, fmt=fmt, eps=eps)
    t = _to_xyxy(tgt_boxes, fmt=fmt, eps=eps)

    pcx, pcy, pw, ph = _center_wh_xyxy(p, eps=eps)
    tcx, tcy, tw, th = _center_wh_xyxy(t, eps=eps)

    w2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2 + ((pw - tw) ** 2) / 4 + ((ph - th) ** 2) / 4
    d = torch.sqrt(w2.clamp_min(0.0) + eps)
    sim = torch.exp(-d / float(max(normalizer, eps)))
    return 1.0 - sim


def wiou_loss_aligned(pred_boxes, tgt_boxes, *, fmt: str = "xyxy", eps: float = 1e-7, distance_weight: float = 1.0):
    """WIoU-style loss for aligned boxes.

    Implements a common WIoU variant:
      loss = (1 - IoU) * exp(k * (rho^2 / c^2))
    where rho^2 is squared center distance, and c^2 is squared diagonal of the smallest enclosing box.

    Returns per-box loss with shape (N,).
    """

    torch = _try_import_torch()

    p = _to_xyxy(pred_boxes, fmt=fmt, eps=eps)
    t = _to_xyxy(tgt_boxes, fmt=fmt, eps=eps)

    iou = _box_iou_xyxy(p, t, eps=eps)

    pcx, pcy, _pw, _ph = _center_wh_xyxy(p, eps=eps)
    tcx, tcy, _tw, _th = _center_wh_xyxy(t, eps=eps)

    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    enc_x1 = torch.minimum(p[:, 0], t[:, 0])
    enc_y1 = torch.minimum(p[:, 1], t[:, 1])
    enc_x2 = torch.maximum(p[:, 2], t[:, 2])
    enc_y2 = torch.maximum(p[:, 3], t[:, 3])
    c2 = (enc_x2 - enc_x1).clamp_min(eps) ** 2 + (enc_y2 - enc_y1).clamp_min(eps) ** 2

    w = torch.exp(float(distance_weight) * (rho2 / c2).clamp_min(0.0))
    return (1.0 - iou) * w


def fwnwd_loss_aligned(pred_boxes, tgt_boxes, *, cfg: FWNWDConfig = FWNWDConfig()):
    """FWNWD loss for aligned boxes.

    Hierarchical composition:
    - Base localization term: WIoU-style loss
    - Focaler term: upweights hard samples using IoU-derived focal factor
    - NWD term: improves sensitivity to small boxes

    Returns either a scalar (reduction != 'none') or (N,) if reduction='none'.
    """

    torch = _try_import_torch()

    p = _to_xyxy(pred_boxes, fmt=cfg.box_format, eps=cfg.eps)
    t = _to_xyxy(tgt_boxes, fmt=cfg.box_format, eps=cfg.eps)

    iou = _box_iou_xyxy(p, t, eps=cfg.eps).clamp(0.0, 1.0)
    focal = (1.0 - iou).clamp_min(0.0) ** float(max(cfg.focal_gamma, 0.0))

    l_wiou = wiou_loss_aligned(
        p,
        t,
        fmt="xyxy",
        eps=cfg.eps,
        distance_weight=cfg.wiou_distance_weight,
    )
    l_nwd = nwd_loss_aligned(p, t, fmt="xyxy", eps=cfg.eps, normalizer=cfg.nwd_normalizer)

    per = l_wiou * focal + float(cfg.nwd_lambda) * l_nwd

    if cfg.reduction == "none":
        return per
    if cfg.reduction == "sum":
        return per.sum()
    if cfg.reduction == "mean":
        return per.mean() if per.numel() else torch.zeros((), dtype=per.dtype, device=per.device)

    raise ValueError(f"Unsupported reduction: {cfg.reduction}")
