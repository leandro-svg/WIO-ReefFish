from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fish-monitoring underwater")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("fwnwd", help="Sanity-check FWNWD loss on random boxes")
    sp.add_argument("--n", type=int, default=128)

    sp = sub.add_parser("udino", help="Sanity-check UDINO refine block on random features")
    sp.add_argument("--b", type=int, default=2)
    sp.add_argument("--c", type=int, default=64)
    sp.add_argument("--h", type=int, default=80)
    sp.add_argument("--w", type=int, default=80)

    return p


def _cmd_fwnwd(n: int) -> int:
    import torch

    from fish_monitoring.underwater.fwnwd_loss import FWNWDConfig, fwnwd_loss_aligned

    # Random boxes in a 640x640 image (xyxy)
    x1y1 = torch.rand((n, 2)) * 600.0
    wh = torch.rand((n, 2)).clamp_min(0.01) * 40.0
    x2y2 = (x1y1 + wh).clamp_max(639.0)
    tgt = torch.cat([x1y1, x2y2], dim=1)

    noise = torch.randn_like(tgt) * 5.0
    pred = (tgt + noise)
    pred[:, 0:2] = pred[:, 0:2].clamp(0.0, 639.0)
    pred[:, 2:4] = pred[:, 2:4].clamp(0.0, 639.0)

    cfg = FWNWDConfig(reduction="mean", focal_gamma=2.0, wiou_distance_weight=1.0, nwd_lambda=1.0)
    loss = fwnwd_loss_aligned(pred, tgt, cfg=cfg)
    print(f"FWNWD loss (mean over {n}): {float(loss):.6f}")
    return 0


def _cmd_udino(b: int, c: int, h: int, w: int) -> int:
    import torch

    from fish_monitoring.underwater.udino_modules import UDINORefineBlock

    x = torch.randn((b, c, h, w), dtype=torch.float32)
    block = UDINORefineBlock(c)
    y = block(x)

    print(f"UDINO refine: in={tuple(x.shape)} out={tuple(y.shape)}")
    print(f"  mean(abs(delta))={float((y - x).abs().mean().detach()):.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "fwnwd":
        return _cmd_fwnwd(args.n)

    if args.cmd == "udino":
        return _cmd_udino(args.b, args.c, args.h, args.w)

    raise AssertionError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
