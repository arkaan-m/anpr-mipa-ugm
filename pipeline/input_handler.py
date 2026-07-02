"""
Stage 1 — Input Handler.

Accepts JPEG/PNG images (≥640×480) and MP4/AVI video clips (≤30s).
For videos, extracts keyframes at 1 FPS via OpenCV.

Returns a list of BGR numpy arrays plus per-frame metadata so downstream
stages can preserve frame_index in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import (
    ALLOWED_IMAGE_EXTS,
    ALLOWED_VIDEO_EXTS,
    KEYFRAME_FPS,
    MAX_VIDEO_SEC,
    MIN_IMG_H,
    MIN_IMG_W,
)


class InputError(ValueError):
    """Raised when an input file is rejected before any pipeline stage runs."""


@dataclass
class InputFrame:
    image: np.ndarray  # BGR
    frame_index: int
    is_video_frame: bool
    timestamp_sec: float = 0.0  # video-only; 0 for stills


def load(filepath: str | Path) -> list[InputFrame]:
    """Dispatch image vs video, return list of frames."""
    path = Path(filepath)
    if not path.exists():
        raise InputError(f"file not found: {path}")

    ext = path.suffix.lower()
    if ext in ALLOWED_IMAGE_EXTS:
        return [_load_image(path)]
    if ext in ALLOWED_VIDEO_EXTS:
        return _load_video(path)
    raise InputError(
        f"unsupported extension '{ext}'. "
        f"images: {sorted(ALLOWED_IMAGE_EXTS)}, videos: {sorted(ALLOWED_VIDEO_EXTS)}"
    )


def _load_image(path: Path) -> InputFrame:
    # cv2.imread ignores the EXIF Orientation tag, so phone photos taken in
    # landscape come back rotated 90° (and YOLO sees the plate sideways).
    # PIL's exif_transpose rotates to the intended viewing orientation.
    try:
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.array(pil_img.convert("RGB"))
    except Exception as e:
        raise InputError(f"PIL could not decode image: {path} ({e})") from e

    if rgb.size == 0:
        raise InputError(f"empty image after decode: {path}")
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if w < MIN_IMG_W or h < MIN_IMG_H:
        raise InputError(
            f"image too small ({w}×{h}); minimum is {MIN_IMG_W}×{MIN_IMG_H}"
        )
    return InputFrame(image=img, frame_index=0, is_video_frame=False)


def _load_video(path: Path) -> list[InputFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(f"OpenCV could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            raise InputError(f"video metadata unreadable (fps={fps}, frames={total})")

        duration = total / fps
        if duration > MAX_VIDEO_SEC:
            raise InputError(
                f"video too long ({duration:.1f}s); maximum is {MAX_VIDEO_SEC}s"
            )

        step = max(int(round(fps / KEYFRAME_FPS)), 1)
        frames: list[InputFrame] = []
        idx = 0
        out_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(
                    InputFrame(
                        image=frame,
                        frame_index=out_idx,
                        is_video_frame=True,
                        timestamp_sec=idx / fps,
                    )
                )
                out_idx += 1
            idx += 1

        if not frames:
            raise InputError("video yielded no frames after sampling")
        return frames
    finally:
        cap.release()
