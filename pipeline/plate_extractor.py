"""
Stage 4 — Plate Cropper & OCR Preprocessor.

Crops the detected bbox from the *original* high-resolution image (with a
5% expansion margin), saves the crop to results/crops/, then applies the
OCR-prep pipeline:
    1. Deskew via Hough line transform (±15° max)
    2. Otsu adaptive binarization
    3. 3×3 median denoising
    4. PIL unsharp mask (radius=1, percent=150, threshold=3)

Returns the processed plate image plus the path to the saved raw crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from config import (
    CROPS_DIR,
    CROP_EXPAND,
    DESKEW_MAX_ANGLE,
    MEDIAN_K,
    UNSHARP_PCT,
    UNSHARP_RADIUS,
    UNSHARP_THRESH,
)
from pipeline import BoundingBox


@dataclass
class PlateCrop:
    raw_bgr: np.ndarray          # cropped color (expanded bbox), pre-deskew
    deskewed_bgr: np.ndarray     # deskewed color — fed to EasyOCR (natural-scene OCR)
    processed_gray: np.ndarray   # deskew + Otsu binarize + median + unsharp — fed to Tesseract
    saved_path: Path


def extract(original_bgr: np.ndarray, bb: BoundingBox, source_name: str) -> PlateCrop:
    raw = _crop_expanded(original_bgr, bb)
    raw = _correct_mirror(raw)
    saved = _save_crop(raw, source_name)

    deskewed = _deskew(raw)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, MEDIAN_K)
    sharpened = _unsharp(denoised)

    return PlateCrop(
        raw_bgr=raw,
        deskewed_bgr=deskewed,
        processed_gray=sharpened,
        saved_path=saved,
    )


def _correct_mirror(crop: np.ndarray) -> np.ndarray:
    """Flip horizontally if the mirrored orientation gives a better plate read."""
    import re as _re
    _PLATE_RE = _re.compile(r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}')

    def _quick_ocr(bgr: np.ndarray) -> str:
        try:
            import easyocr
            if not hasattr(_correct_mirror, '_reader'):
                _correct_mirror._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            texts = _correct_mirror._reader.readtext(rgb, detail=0)
            raw = ' '.join(texts).upper()
            return _re.sub(r'[^A-Z0-9 ]', '', raw).strip()
        except Exception:
            return ''

    text_normal = _quick_ocr(crop)
    flipped = cv2.flip(crop, 1)
    text_flipped = _quick_ocr(flipped)

    def _score(t: str) -> int:
        if _PLATE_RE.search(t):
            return 2
        if _re.search(r'[A-Z]', t) and _re.search(r'\d', t):
            return 1
        return 0

    if _score(text_flipped) > _score(text_normal):
        return flipped
    return crop


def _crop_expanded(img: np.ndarray, bb: BoundingBox) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int(round(bb.width * CROP_EXPAND))
    dy = int(round(bb.height * CROP_EXPAND))
    x1 = max(0, bb.x1 - dx)
    y1 = max(0, bb.y1 - dy)
    x2 = min(w, bb.x2 + dx)
    y2 = min(h, bb.y2 + dy)
    return img[y1:y2, x1:x2].copy()


def _save_crop(crop: np.ndarray, source_name: str) -> Path:
    stem = Path(source_name).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = CROPS_DIR / f"{stem}_{ts}.png"
    cv2.imwrite(str(out), crop)
    return out


def _deskew(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return bgr

    angles: list[float] = []
    for rho_theta in lines[:30]:
        theta = float(rho_theta[0][1])
        deg = np.degrees(theta) - 90.0  # horizontal lines → 0°
        if -DESKEW_MAX_ANGLE <= deg <= DESKEW_MAX_ANGLE:
            angles.append(deg)
    if not angles:
        return bgr

    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return bgr

    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _unsharp(gray: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(gray)
    sharp = pil.filter(
        ImageFilter.UnsharpMask(
            radius=UNSHARP_RADIUS, percent=UNSHARP_PCT, threshold=UNSHARP_THRESH,
        )
    )
    return np.array(sharp)
