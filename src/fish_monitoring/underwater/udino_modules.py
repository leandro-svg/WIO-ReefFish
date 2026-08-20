from __future__ import annotations

from dataclasses import dataclass


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for fish_monitoring.underwater") from e


try:  # pragma: no cover
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass(frozen=True)
class FrequencyEnhanceConfig:
    """Config for the UDINO-style frequency enhancement.

    radii: fractions of the max frequency radius (0..1) used as cutoffs.
    """

    radii: tuple[float, ...] = (0.15, 0.30, 0.45)
    learnable_scale: bool = True
    eps: float = 1e-6


class MultiScaleHighFrequencyEnhance(nn.Module):
    """Multi-scale high-frequency enhancement in the frequency domain.

    Input/Output: (B, C, H, W)

    This block computes an FFT per channel, applies a high-pass mask at multiple radii,
    converts back via iFFT, and adds the enhanced residual to the input.
    """

    def __init__(self, channels: int, cfg: FrequencyEnhanceConfig = FrequencyEnhanceConfig()):
        if nn is None or torch is None:  # pragma: no cover
            _try_import_torch()
        super().__init__()
        self.channels = int(channels)
        self.cfg = cfg

        # Initialize as near-identity (alpha=0) to avoid destabilizing training at step 0.
        init = torch.zeros((len(cfg.radii),), dtype=torch.float32)
        if cfg.learnable_scale:
            self.alpha = nn.Parameter(init)
        else:
            self.register_buffer("alpha", init, persistent=False)

    def _radial_grid(self, H: int, W: int, device, dtype):
        fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype)  # rfft2 keeps only non-negative x freqs
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(yy * yy + xx * xx)
        r = r / (r.max().clamp_min(self.cfg.eps))
        return r  # (H, W//2+1)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got shape {tuple(x.shape)}")

        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={C}")

        # Compute enhancement from a detached view to avoid backprop through FFT,
        # which can be numerically fragile under AMP+GradScaler.
        x_det = x.detach()

        # FFT over spatial dims
        X = torch.fft.rfft2(x_det, dim=(-2, -1), norm="ortho")
        r = self._radial_grid(H, W, device=x.device, dtype=x.real.dtype)

        enh = 0.0
        for i, rad in enumerate(self.cfg.radii):
            cutoff = float(rad)
            mask = (r >= cutoff).to(X.dtype)
            Xi = X * mask
            xi = torch.fft.irfft2(Xi, s=(H, W), dim=(-2, -1), norm="ortho")
            # Bound contribution to keep training stable.
            enh = enh + torch.tanh(self.alpha[i]) * xi

        return x + enh


class GatedChannelRefine(nn.Module):
    """Gated channel refinement (SE-like) used in UDINO-style refinement."""

    def __init__(self, channels: int, reduction: int = 16):
        if nn is None:  # pragma: no cover
            _try_import_torch()
        super().__init__()
        r = max(1, int(reduction))
        hidden = max(1, channels // r)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Gate strength starts at 0 => exact identity.
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        # Use detached features to avoid sending gradients back through the gate
        # into the upstream graph (UDINO is a preprocessor; this keeps training stable).
        g = self.pool(x.detach())
        w = self.mlp(g)
        # Start as identity and learn a bounded deviation from it.
        # When beta=0 => x. When beta>0 => emphasize channels; beta<0 => suppress.
        scale = 1.0 + torch.tanh(self.beta) * (2.0 * w - 1.0)
        return x * scale


class UDINORefineBlock(nn.Module):
    """Convenience block: frequency enhancement + gated channel refinement."""

    def __init__(
        self,
        channels: int,
        *,
        freq_cfg: FrequencyEnhanceConfig = FrequencyEnhanceConfig(),
        gate_reduction: int = 16,
    ):
        if nn is None:  # pragma: no cover
            _try_import_torch()
        super().__init__()
        self.freq = MultiScaleHighFrequencyEnhance(channels, cfg=freq_cfg)
        self.gate = GatedChannelRefine(channels, reduction=gate_reduction)

    def forward(self, x):
        x = self.freq(x)
        x = self.gate(x)
        return x
