"""
ANPR pipeline package — core data types.

These are the canonical structs that flow between stages 1–10. All stage
modules import from here; nothing else defines these names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Status enum (string-valued for CSV/log readability)
STATUS_AUTHORIZED = "AUTHORIZED"
STATUS_UNAUTHORIZED = "UNAUTHORIZED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_OCR_FAILED = "OCR_FAILED"
STATUS_NO_PLATE = "NO_PLATE_FOUND"

# Error categories for the verification log
ERR_DETECTION_FAILURE = "DETECTION_FAILURE"
ERR_OCR_FAILURE = "OCR_FAILURE"
ERR_DB_MISMATCH = "DB_MISMATCH"
ERR_SUCCESS = "SUCCESS"


class ValidationError(ValueError):
    """Raised by text_normalizer when a plate fails structural validation."""


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class OcrOutput:
    text: str
    confidence: float
    engine: str  # "easyocr" | "tesseract"
    is_valid: bool = False  # passes one of the plate regexes


@dataclass
class VerificationResult:
    status: str
    matched_plate: Optional[str] = None
    match_type: str = "NONE"  # EXACT | FUZZY | NONE
    ocr_confidence: float = 0.0
    detection_confidence: float = 0.0
    flag: Optional[str] = None  # e.g. "FUZZY_MATCH - verify manually"


@dataclass
class PlateOutcome:
    """One detected plate, fully processed through OCR + verification."""
    bbox: BoundingBox
    crop_image: Optional[np.ndarray] = None
    crop_path: Optional[str] = None
    easyocr: Optional[OcrOutput] = None
    tesseract: Optional[OcrOutput] = None
    chosen_engine: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    ocr_confidence: float = 0.0
    verification: Optional[VerificationResult] = None


@dataclass
class PipelineResult:
    """End-to-end result for a single image (or one video keyframe)."""
    source_path: str
    frame_index: int = 0
    is_video_frame: bool = False
    plates: list[PlateOutcome] = field(default_factory=list)
    annotated_path: Optional[str] = None
    processing_time_ms: float = 0.0
    error_category: str = ERR_SUCCESS
    error_message: Optional[str] = None
    status: str = STATUS_NO_PLATE  # rolled-up worst/best status across plates
