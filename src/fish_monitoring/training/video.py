"""Video rendering with detection overlays.

Generates annotated videos from image sequences with bounding-box
predictions drawn on each frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


BRIGHT_COLORS: Sequence[tuple[int, int, int]] = (
    (0, 255, 255),
    (0, 128, 255),
    (255, 51, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (153, 0, 255),
    (0, 204, 204),
    (0, 153, 255),
)


@dataclass(frozen=True)
class VideoArgs:
    weights: Path
    input_video: Path
    output_dir: Path
    conf_thr: float = 0.25
    tracker_cfg: str = "bytetrack.yaml"
    seconds: float | None = None
    device: int | str = 0
    crf: int = 18
    preset: str = "fast"


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError("opencv-python is not installed or failed to import") from e


def _import_ultralytics_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ultralytics is not installed or failed to import") from e


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return BRIGHT_COLORS[int(track_id) % len(BRIGHT_COLORS)]


def get_contrast_text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 186 else (255, 255, 255)


def draw_labelled_box(
    image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: tuple[int, int, int],
    font_scale: float = 1.3,
    thickness_outline: int = 3,
    thickness_text: int = 2,
    box_thickness: int = 4,
) -> None:
    cv2 = _import_cv2()

    h, w = image.shape[:2]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w - 1, int(x2)), min(h - 1, int(y2))

    cv2.rectangle(image, (x1i, y1i), (x2i, y2i), color, box_thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
    text_x = max(x1i, 0)
    text_y = max(y1i - 7, th + 5)

    text_color = get_contrast_text_color(color)
    cv2.rectangle(image, (text_x, text_y - th - 4), (text_x + tw + 6, text_y), color, -1)

    cv2.putText(
        image,
        label,
        (text_x + 3, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness=thickness_outline,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (text_x + 3, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness=thickness_text,
        lineType=cv2.LINE_AA,
    )


def render_video_with_tracking(
    *,
    model_weights: Path,
    input_path: Path,
    temp_out_path: Path,
    seconds: float | None,
    conf_thr: float,
    device: int | str,
    tracker_cfg: str,
) -> None:
    cv2 = _import_cv2()
    YOLO = _import_ultralytics_yolo()

    model = YOLO(str(model_weights))
    class_names = [model.names[i] for i in range(len(model.names))]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    max_frames = total_frames
    if seconds is not None:
        max_frames = min(total_frames, int(round(seconds * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.track(
            source=frame,
            conf=conf_thr,
            tracker=tracker_cfg,
            persist=True,
            device=device,
            verbose=False,
            stream=False,
        )[0]

        if res.boxes is not None and len(res.boxes) > 0:
            ids = res.boxes.id
            for i, box in enumerate(res.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cname = class_names[cls] if cls < len(class_names) else str(cls)

                tid = None
                if ids is not None:
                    tid = int(ids[i].item() if hasattr(ids[i], "item") else int(ids[i]))

                label = f"{cname} {conf:.2f}"
                color = color_for_id(tid) if tid is not None else BRIGHT_COLORS[cls % len(BRIGHT_COLORS)]
                draw_labelled_box(frame, x1, y1, x2, y2, label, color)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def ffmpeg_reencode_h264(src_path: Path, dst_path: Path, crf: int = 18, preset: str = "medium") -> None:
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)


def run_video(args: VideoArgs) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_mp4 = args.output_dir / f"tmp_pred_{timestamp}.mp4"
    final_mp4 = args.output_dir / f"pred_{timestamp}.mp4"

    render_video_with_tracking(
        model_weights=args.weights,
        input_path=args.input_video,
        temp_out_path=temp_mp4,
        seconds=args.seconds,
        conf_thr=args.conf_thr,
        device=args.device,
        tracker_cfg=args.tracker_cfg,
    )

    ffmpeg_reencode_h264(temp_mp4, final_mp4, crf=args.crf, preset=args.preset)

    try:
        temp_mp4.unlink(missing_ok=True)
    except Exception:
        pass

    return final_mp4
