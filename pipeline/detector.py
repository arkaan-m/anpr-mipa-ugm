"""
Stage 3 — YOLOv8 License Plate Detector.

Loads a fine-tuned YOLOv8 weight, runs inference on the letterboxed RGB
frame from Stage 2, then maps boxes back to original-image coordinates
and applies post-detection filtering (aspect ratio, min area).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    ASPECT_MAX,
    ASPECT_MIN,
    DEFAULT_MODEL_VARIANT,
    DET_CONF,
    INPUT_SIZE,
    MIN_BBOX_AREA,
    MODELS_DIR,
    MODEL_VARIANTS,
    NMS_IOU,
)
from pipeline import BoundingBox
from pipeline.preprocessor import PreprocessedFrame


class DetectorLoadError(RuntimeError):
    """Raised when the YOLOv8 weight file cannot be found or loaded."""


_MODEL_CACHE: dict[str, object] = {}


def _resolve_weight_path(variant: str) -> Path:
    if variant not in MODEL_VARIANTS:
        raise DetectorLoadError(
            f"unknown variant '{variant}'. options: {list(MODEL_VARIANTS)}"
        )
    fname = MODEL_VARIANTS[variant]
    for p in (MODELS_DIR / fname, MODELS_DIR.parent / fname):
        if p.exists():
            return p
    raise DetectorLoadError(
        f"weight file '{fname}' not found in {MODELS_DIR} or project root"
    )


def load_model(variant: str = DEFAULT_MODEL_VARIANT):
    """Lazy-load and cache the YOLOv8 model. `variant` is yolov8n/s/m."""
    if variant in _MODEL_CACHE:
        return _MODEL_CACHE[variant]
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise DetectorLoadError("ultralytics not installed; pip install ultralytics") from e
    weight = _resolve_weight_path(variant)
    model = YOLO(str(weight))
    _MODEL_CACHE[variant] = model
    return model


def detect(
    pre: PreprocessedFrame,
    variant: str = DEFAULT_MODEL_VARIANT,
    model: Optional[object] = None,
) -> list[BoundingBox]:
    """Run inference; return BoundingBoxes in *original* image coords.

    If no plate is found at full-image scale, retries on the bottom 60% of
    the original image (where plates almost always appear). This recovers
    wide-angle whole-car shots where the plate is too small at full scale.
    """
    m = model if model is not None else load_model(variant)
    boxes = _run_inference(m, pre.image_rgb, pre.scale, pre.pad_x, pre.pad_y, pre.original_bgr.shape)
    if boxes:
        return boxes

    # Fallback: zoom into bottom 60% of original, re-letterbox, retry
    import cv2
    orig = pre.original_bgr
    orig_h, orig_w = orig.shape[:2]
    crop_y = int(orig_h * 0.40)
    bottom_crop = orig[crop_y:, :]

    from pipeline.preprocessor import _letterbox
    lb, scale2, pad_x2, pad_y2 = _letterbox(bottom_crop, INPUT_SIZE)
    rgb2 = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

    # Boxes come back in bottom-crop coords; shift y back to full-image coords
    crop_boxes = _run_inference(m, rgb2, scale2, pad_x2, pad_y2,
                                bottom_crop.shape, y_offset=crop_y)
    return crop_boxes


def _run_inference(
    m,
    image_rgb: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_shape: tuple,
    y_offset: int = 0,
) -> list[BoundingBox]:
    results = m.predict(
        source=image_rgb,
        conf=DET_CONF,
        iou=NMS_IOU,
        imgsz=INPUT_SIZE[0],
        verbose=False,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    orig_h, orig_w = orig_shape[:2]
    out: list[BoundingBox] = []
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        ox1, oy1, ox2, oy2 = _unletterbox((x1, y1, x2, y2), scale, pad_x, pad_y)
        ox1 = int(max(0, min(orig_w - 1, ox1)))
        oy1 = int(max(0, min(orig_h - 1, oy1 + y_offset)))
        ox2 = int(max(0, min(orig_w - 1, ox2)))
        oy2 = int(max(0, min(orig_h - 1, oy2 + y_offset)))
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        bb = BoundingBox(x1=ox1, y1=oy1, x2=ox2, y2=oy2, confidence=float(conf))
        if not _passes_filters(bb):
            continue
        out.append(bb)
    return out


def _unletterbox(
    xyxy: tuple[float, float, float, float],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )


def _passes_filters(bb: BoundingBox) -> bool:
    if bb.area < MIN_BBOX_AREA:
        return False
    if not (ASPECT_MIN <= bb.aspect_ratio <= ASPECT_MAX):
        return False
    return True
