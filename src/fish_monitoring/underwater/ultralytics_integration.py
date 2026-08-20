from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _import_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for Ultralytics integration") from e


@dataclass(frozen=True)
class UnderwaterTrainConfig:
    """Optional patches applied to Ultralytics YOLO training."""

    use_udino: bool = False
    udino_radii: tuple[float, ...] = (0.15, 0.30, 0.45)
    udino_gate_reduction: int = 16

    use_fwnwd: bool = False
    fwnwd_focal_gamma: float = 2.0
    fwnwd_wiou_distance_weight: float = 1.0
    fwnwd_nwd_lambda: float = 1.0
    fwnwd_normalizer: float = 200.0


def inject_udino_preprocessor(det_model, *, radii: Iterable[float] = (0.15, 0.30, 0.45), gate_reduction: int = 16) -> None:
    """Prepend UDINORefineBlock to an Ultralytics DetectionModel.

    det_model is expected to be `ultralytics.nn.tasks.DetectionModel`.

    This keeps the original YOLO weights intact and adds a small learnable
    pre-processing block that is trained jointly.
    """

    _torch, _nn = _import_torch()

    # Avoid importing torch-dependent modules unless needed.
    from fish_monitoring.underwater.udino_modules import FrequencyEnhanceConfig, UDINORefineBlock

    if getattr(det_model, "_fishmon_udino_injected", False):
        return

    apply_udino_predict_patch()

    # Register module so its parameters are optimized.
    freq_cfg = FrequencyEnhanceConfig(radii=tuple(float(x) for x in radii))
    det_model.fishmon_udino = UDINORefineBlock(3, freq_cfg=freq_cfg, gate_reduction=int(gate_reduction))  # type: ignore[attr-defined]

    # Keep device consistent with the host model.
    try:
        import torch.nn as nn

        if isinstance(det_model, nn.Module):
            try:
                dev = next(det_model.parameters()).device
                det_model.fishmon_udino.to(dev)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    det_model._fishmon_udino_injected = True


def apply_udino_predict_patch() -> None:
    """Patch Ultralytics DetectionModel.predict to apply UDINO when present.

    This makes UDINO run during training/val/infer without modifying the internal
    parsed model graph, and it also keeps behavior unchanged for base models.
    """

    import ultralytics.nn.tasks as ul_tasks

    if getattr(ul_tasks.DetectionModel, "_fishmon_udino_predict_patched", False):
        return

    orig_predict = ul_tasks.DetectionModel.predict

    def _predict(self, x, *args, **kwargs):
        if hasattr(self, "fishmon_udino"):
            # Ultralytics uses AMP (float16 inputs). Our UDINO block includes FFT and
            # conv layers that can hit dtype/bias mismatches under some AMP paths.
            # Run UDINO in float32 then cast back.
            orig_dtype = getattr(x, "dtype", None)
            x_fp32 = x.float()

            # Force UDINO to run without autocast to reduce numerical issues.
            try:
                import torch

                with torch.autocast(device_type=str(x_fp32.device.type), enabled=False):
                    x_fp32 = self.fishmon_udino(x_fp32)
            except Exception:
                x_fp32 = self.fishmon_udino(x_fp32)

            # Safety: prevent NaNs/Infs from poisoning training.
            try:
                import torch

                if not torch.isfinite(x_fp32).all():
                    # Fallback to original input for this call.
                    x_fp32 = x.float()
                else:
                    x_fp32 = torch.nan_to_num(x_fp32, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception:
                pass

            # Images are expected in [0, 1] after Ultralytics preprocessing.
            # Clamp to keep UDINO from shifting the input distribution too far.
            try:
                x_fp32 = x_fp32.clamp_(0.0, 1.0)
            except Exception:
                x_fp32 = x_fp32.clamp(0.0, 1.0)

            if orig_dtype is not None:
                x = x_fp32.to(orig_dtype)
            else:
                x = x_fp32
        return orig_predict(self, x, *args, **kwargs)

    ul_tasks.DetectionModel.predict = _predict  # type: ignore[assignment]
    ul_tasks.DetectionModel._fishmon_udino_predict_patched = True
    ul_tasks.DetectionModel._fishmon_udino_predict_orig = orig_predict


def apply_fwnwd_loss_patch(
    *,
    focal_gamma: float = 2.0,
    wiou_distance_weight: float = 1.0,
    nwd_lambda: float = 1.0,
    nwd_normalizer: float = 200.0,
) -> None:
    """Monkeypatch Ultralytics `BboxLoss.forward` to use FWNWD.

    Ultralytics v8.3.x constructs `v8DetectionLoss`, which creates `BboxLoss(...)`.
    Patching the class method before `model.train(...)` ensures training uses this loss.

    This replaces the IoU term (CIoU) with:
      FWNWD = focal(IoU) * WIoU(pred, tgt) + lambda * NWD(pred, tgt)

    DFL loss remains unchanged.
    """

    torch, _nn = _import_torch()

    import ultralytics.utils.loss as ul_loss

    if getattr(ul_loss.BboxLoss, "_fishmon_fwnwd_patched", False):
        return

    from fish_monitoring.underwater.fwnwd_loss import FWNWDConfig, fwnwd_loss_aligned

    orig_forward = ul_loss.BboxLoss.forward

    def _patched_forward(
        self,
        pred_dist: Any,
        pred_bboxes: Any,
        anchor_points: Any,
        target_bboxes: Any,
        target_scores: Any,
        target_scores_sum: Any,
        fg_mask: Any,
    ):
        # Keep empty-foreground behavior stable.
        if fg_mask is None or fg_mask.sum() == 0:
            loss_iou = torch.tensor(0.0, device=pred_dist.device)
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
            return loss_iou, loss_dfl

        # Original weighting
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        cfg = FWNWDConfig(
            box_format="xyxy",
            reduction="none",
            focal_gamma=float(focal_gamma),
            wiou_distance_weight=float(wiou_distance_weight),
            nwd_lambda=float(nwd_lambda),
            nwd_normalizer=float(nwd_normalizer),
        )

        per = fwnwd_loss_aligned(pred_bboxes[fg_mask], target_bboxes[fg_mask], cfg=cfg)  # (Nfg,)
        loss_iou = (per.unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL loss (unchanged from Ultralytics)
        if self.dfl_loss:
            target_ltrb = ul_loss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    ul_loss.BboxLoss.forward = _patched_forward  # type: ignore[assignment]
    ul_loss.BboxLoss._fishmon_fwnwd_patched = True
    ul_loss.BboxLoss._fishmon_fwnwd_orig_forward = orig_forward
