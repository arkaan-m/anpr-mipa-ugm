"""
Stage 2 — Preprocessor.

Letterbox to 640×640 (gray padding, aspect preserved), apply CLAHE in LAB
space, gate Wiener deconvolution on Laplacian-variance blur score,
return (preprocessed_RGB, original_BGR_for_cropping).

The original (full-resolution, untouched) image is also returned so that
plate_extractor can crop from the high-res source rather than the
640×640 letterboxed copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import wiener

from config import (
    BLUR_LAP_THRESH,
    CLAHE_CLIP,
    CLAHE_TILE,
    ENABLE_CLAHE,
    ENABLE_WIENER,
    INPUT_SIZE,
    WIENER_NOISE_VAR,
)


@dataclass
class PreprocessedFrame:
    """`image_rgb` is what YOLOv8 sees. `original_bgr` is untouched source.
    `scale`/`pad_x`/`pad_y` let us map YOLO bbox coords back to the original.
    """
    image_rgb: np.ndarray
    original_bgr: np.ndarray
    scale: float
    pad_x: int
    pad_y: int
    blur_score: float
    deblurred: bool


def preprocess(bgr: np.ndarray) -> PreprocessedFrame:
    original = bgr.copy()
    work = bgr

    blur_score = _laplacian_variance(work)
    deblurred = False
    if ENABLE_WIENER and blur_score < BLUR_LAP_THRESH:
        work = _wiener_deblur(work)
        deblurred = True

    if ENABLE_CLAHE:
        work = _clahe_lab(work)

    letterboxed, scale, pad_x, pad_y = _letterbox(work, INPUT_SIZE)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    return PreprocessedFrame(
        image_rgb=rgb,
        original_bgr=original,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        blur_score=float(blur_score),
        deblurred=deblurred,
    )


def _laplacian_variance(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=tuple(CLAHE_TILE))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def _wiener_deblur(bgr: np.ndarray) -> np.ndarray:
    out = np.empty_like(bgr)
    for c in range(3):
        chan = bgr[:, :, c].astype(np.float64)
        filtered = wiener(chan, mysize=5, noise=WIENER_NOISE_VAR)
        out[:, :, c] = np.clip(np.nan_to_num(filtered), 0, 255).astype(np.uint8)
    return out


def _letterbox(bgr: np.ndarray, target: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    """Resize with aspect preserved, pad with gray (114). Returns (img, scale, pad_x, pad_y)."""
    tw, th = target
    h, w = bgr.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y
