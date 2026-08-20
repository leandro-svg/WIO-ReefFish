"""YOLO / Ultralytics training and validation operations.

Wraps the Ultralytics API for training, validation, and system checks,
with optional underwater-domain augmentation (FWNWD loss, U-DINO).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fish_monitoring.underwater.ultralytics_integration import UnderwaterTrainConfig, apply_fwnwd_loss_patch, inject_udino_preprocessor


def _import_ultralytics():
    try:
        import ultralytics  # type: ignore
        from ultralytics import YOLO  # type: ignore

        return ultralytics, YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is not installed or failed to import. "
            "Install it (e.g. `pip install ultralytics`) and ensure torch/cuda is set up."
        ) from e


def checks() -> None:
    ultralytics, _YOLO = _import_ultralytics()
    ultralytics.checks()


@dataclass(frozen=True)
class TrainArgs:
    data_yaml: Path
    weights: str
    epochs: int = 100
    imgsz: int = 640
    batch: int = 32
    device: Any = 0
    fraction: float = 1.0
    val: bool = True
    patience: int = 20
    project: str = "results"
    name: str = "run"
    underwater: UnderwaterTrainConfig = UnderwaterTrainConfig()


def train(args: TrainArgs, traceback_path: Path | None = None) -> None:
    """Train YOLO with optional traceback capture."""
    import traceback

    _ultralytics, YOLO = _import_ultralytics()
    model = YOLO(args.weights)

    # Optional underwater patches (kept off by default).
    if args.underwater.use_fwnwd:
        apply_fwnwd_loss_patch(
            focal_gamma=args.underwater.fwnwd_focal_gamma,
            wiou_distance_weight=args.underwater.fwnwd_wiou_distance_weight,
            nwd_lambda=args.underwater.fwnwd_nwd_lambda,
            nwd_normalizer=args.underwater.fwnwd_normalizer,
        )
    if args.underwater.use_udino:
        # Ultralytics Trainer builds/owns the actual nn.Module. Inject via callbacks.
        def _attach_udino(trainer, *a, **k):
            try:
                import torch.nn as nn
            except Exception:
                return

            tm = getattr(trainer, "model", None)
            if isinstance(tm, nn.Module):
                inject_udino_preprocessor(
                    tm,
                    radii=args.underwater.udino_radii,
                    gate_reduction=args.underwater.udino_gate_reduction,
                )

            ema = getattr(trainer, "ema", None)
            ema_model = getattr(ema, "ema", None)
            if isinstance(ema_model, nn.Module):
                inject_udino_preprocessor(
                    ema_model,
                    radii=args.underwater.udino_radii,
                    gate_reduction=args.underwater.udino_gate_reduction,
                )

            # One-time confirmation in logs.
            if not getattr(trainer, "_fishmon_udino_logged", False):
                print(
                    "FishMon: UDINO enabled (attached to trainer.model"
                    + (" + EMA" if isinstance(ema_model, nn.Module) else "")
                    + ")"
                )
                trainer._fishmon_udino_logged = True

        model.add_callback("on_pretrain_routine_start", _attach_udino)
        model.add_callback("on_train_start", _attach_udino)

        # UDINO is experimental and has shown NaNs under AMP on some runs.
        # Train in full precision to keep loss stable.
        force_amp = False
        print("FishMon: UDINO enabled -> forcing amp=False to avoid NaNs")
    else:
        force_amp = None

    try:
        model.train(
            data=str(args.data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            fraction=float(args.fraction),
            val=args.val,
            patience=args.patience,
            project=args.project,
            name=args.name,
            **({"amp": force_amp} if force_amp is not None else {}),
        )
    except Exception:
        if traceback_path is not None:
            traceback_path.parent.mkdir(parents=True, exist_ok=True)
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"Traceback written to: {traceback_path}")
        raise


def train_with_fallback(
    primary: TrainArgs,
    fallback: TrainArgs,
    traceback_path: Path,
    primary_header: str = "=== PRIMARY ATTEMPT ===",
    fallback_header: str = "=== FALLBACK ATTEMPT ===",
    sleep_seconds: float = 2.0,
) -> None:
    import time
    import traceback

    _ultralytics, YOLO = _import_ultralytics()
    model = YOLO(primary.weights)

    # Optional underwater patches (kept off by default).
    # Apply once to the shared model before attempts.
    uw = primary.underwater
    if uw.use_fwnwd:
        apply_fwnwd_loss_patch(
            focal_gamma=uw.fwnwd_focal_gamma,
            wiou_distance_weight=uw.fwnwd_wiou_distance_weight,
            nwd_lambda=uw.fwnwd_nwd_lambda,
            nwd_normalizer=uw.fwnwd_normalizer,
        )
    if uw.use_udino:
        def _attach_udino(trainer, *a, **k):
            try:
                import torch.nn as nn
            except Exception:
                return

            tm = getattr(trainer, "model", None)
            if isinstance(tm, nn.Module):
                inject_udino_preprocessor(tm, radii=uw.udino_radii, gate_reduction=uw.udino_gate_reduction)

            ema = getattr(trainer, "ema", None)
            ema_model = getattr(ema, "ema", None)
            if isinstance(ema_model, nn.Module):
                inject_udino_preprocessor(ema_model, radii=uw.udino_radii, gate_reduction=uw.udino_gate_reduction)

            if not getattr(trainer, "_fishmon_udino_logged", False):
                print(
                    "FishMon: UDINO enabled (attached to trainer.model"
                    + (" + EMA" if isinstance(ema_model, nn.Module) else "")
                    + ")"
                )
                trainer._fishmon_udino_logged = True

        model.add_callback("on_pretrain_routine_start", _attach_udino)
        model.add_callback("on_train_start", _attach_udino)

        force_amp = False
        print("FishMon: UDINO enabled -> forcing amp=False to avoid NaNs")
    else:
        force_amp = None

    try:
        model.train(
            data=str(primary.data_yaml),
            epochs=primary.epochs,
            imgsz=primary.imgsz,
            batch=primary.batch,
            device=primary.device,
            fraction=float(primary.fraction),
            val=primary.val,
            patience=primary.patience,
            project=primary.project,
            name=primary.name,
            **({"amp": force_amp} if force_amp is not None else {}),
        )
        return
    except Exception:
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text(primary_header + "\n" + traceback.format_exc(), encoding="utf-8")
        print(f"Primary training failed. Traceback written to: {traceback_path}")

    time.sleep(sleep_seconds)

    try:
        model.train(
            data=str(fallback.data_yaml),
            epochs=fallback.epochs,
            imgsz=fallback.imgsz,
            batch=fallback.batch,
            device=fallback.device,
            fraction=float(fallback.fraction),
            val=fallback.val,
            patience=fallback.patience,
            project=fallback.project,
            name=fallback.name,
            **({"amp": force_amp} if force_amp is not None else {}),
        )
    except Exception:
        with traceback_path.open("a", encoding="utf-8") as f:
            f.write("\n\n" + fallback_header + "\n")
            f.write(traceback.format_exc())
        print(f"Fallback training also failed. Tracebacks appended to: {traceback_path}")
        raise


@dataclass(frozen=True)
class ValArgs:
    model_path: Path
    data_yaml: Path
    split: str = "val"
    imgsz: int = 640
    save_json: bool = False
    plots: bool = True
    verbose: bool = True
    project: str = "results"
    name: str = "eval"


def validate(args: ValArgs):
    _ultralytics, YOLO = _import_ultralytics()
    model = YOLO(str(args.model_path))

    return model.val(
        data=str(args.data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        save_json=args.save_json,
        plots=args.plots,
        verbose=args.verbose,
        project=args.project,
        name=args.name,
    )


def print_cuda_info() -> None:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is not installed or failed to import") from e

    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        reserved_mem = torch.cuda.memory_reserved(i) / 1024**3
        allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
        free_mem = total_mem - reserved_mem
        print(f"GPU {i}: {name}")
        print(f"  total:     {total_mem:.1f} GB")
        print(f"  free(est): {free_mem:.1f} GB")
        print(f"  allocated: {allocated_mem:.1f} GB")
