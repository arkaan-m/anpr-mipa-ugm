"""
Stage 9 — Pipeline Orchestrator.

Ties Stages 1–8 into:
    process_image(filepath) -> list[PipelineResult]   (one per frame; usually 1)
    process_batch(directory) -> list[PipelineResult]
    print_summary(results)

Per-stage exceptions are caught so one bad frame can't kill a batch run.
Each PipelineResult carries its own error_category for forensic review.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from pipeline import (
    ERR_DETECTION_FAILURE,
    ERR_OCR_FAILURE,
    ERR_SUCCESS,
    PipelineResult,
    PlateOutcome,
    STATUS_AUTHORIZED,
    STATUS_NO_PLATE,
    STATUS_OCR_FAILED,
    STATUS_UNAUTHORIZED,
    STATUS_UNCERTAIN,
    ValidationError,
)
from pipeline import input_handler
from pipeline import preprocessor as prep
from pipeline import detector
from pipeline import plate_extractor
from pipeline import ocr_engine
from pipeline import text_normalizer
from pipeline import verifier
from pipeline import logger as result_logger
from config import ALLOWED_IMAGE_EXTS, ALLOWED_VIDEO_EXTS, DEFAULT_MODEL_VARIANT


def process_image(filepath: str | Path, variant: str = DEFAULT_MODEL_VARIANT) -> list[PipelineResult]:
    """Run the full pipeline on one upload. Returns one result per frame
    (1 for stills, N for videos)."""
    path = Path(filepath)
    results: list[PipelineResult] = []

    # Stage 1: load
    try:
        frames = input_handler.load(path)
    except Exception as e:  # noqa: BLE001
        r = PipelineResult(
            source_path=str(path),
            status=STATUS_NO_PLATE,
            error_category=ERR_DETECTION_FAILURE,
            error_message=f"input load failed: {e}",
        )
        result_logger.log(r)
        return [r]

    for frame in frames:
        t0 = time.perf_counter()
        result = PipelineResult(
            source_path=str(path),
            frame_index=frame.frame_index,
            is_video_frame=frame.is_video_frame,
        )

        try:
            # Stage 2: preprocess
            pre = prep.preprocess(frame.image)

            # Stage 3: detect
            boxes = detector.detect(pre, variant=variant)
            if not boxes:
                result.status = STATUS_NO_PLATE
                result.error_category = ERR_DETECTION_FAILURE
            else:
                # Stages 4–7 per detected box
                for bb in boxes:
                    outcome = _process_plate(pre.original_bgr, bb, path.name)
                    result.plates.append(outcome)
                result.status = _roll_up_status(result.plates)
                result.error_category = result_logger.category_for(result.status)

        except Exception as e:  # noqa: BLE001
            result.status = STATUS_NO_PLATE
            result.error_category = ERR_DETECTION_FAILURE
            result.error_message = f"pipeline error: {e}"

        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        result_logger.log(result)
        results.append(result)

    return results


def _process_plate(original_bgr, bb, source_name: str) -> PlateOutcome:
    outcome = PlateOutcome(bbox=bb)

    # Stage 4
    crop = plate_extractor.extract(original_bgr, bb, source_name)
    outcome.crop_image = crop.raw_bgr
    outcome.crop_path = str(crop.saved_path)

    # Stage 5 — EasyOCR sees the deskewed color crop, Tesseract sees the binarized one.
    chosen, easy, tess = ocr_engine.run(crop.deskewed_bgr, crop.processed_gray)
    outcome.easyocr = easy
    outcome.tesseract = tess
    outcome.chosen_engine = chosen.engine
    outcome.raw_text = chosen.text
    outcome.ocr_confidence = chosen.confidence

    if not chosen.is_valid:
        # Stage 6 won't be attempted; verifier still runs with empty plate to log the failure.
        outcome.normalized_text = ""
        from pipeline import VerificationResult
        outcome.verification = VerificationResult(
            status=STATUS_OCR_FAILED,
            ocr_confidence=chosen.confidence,
            detection_confidence=bb.confidence,
        )
        return outcome

    # Stage 6
    try:
        outcome.normalized_text = text_normalizer.normalize(chosen.text)
    except ValidationError:
        outcome.normalized_text = ""
        from pipeline import VerificationResult
        outcome.verification = VerificationResult(
            status=STATUS_OCR_FAILED,
            ocr_confidence=chosen.confidence,
            detection_confidence=bb.confidence,
        )
        return outcome

    # Stage 7
    outcome.verification = verifier.verify(
        outcome.normalized_text,
        ocr_confidence=chosen.confidence,
        detection_confidence=bb.confidence,
    )
    return outcome


def _roll_up_status(plates: list[PlateOutcome]) -> str:
    """If any plate is AUTHORIZED → AUTHORIZED. Else worst of UNCERTAIN/UNAUTHORIZED/OCR_FAILED."""
    if not plates:
        return STATUS_NO_PLATE
    statuses = [p.verification.status for p in plates if p.verification]
    if not statuses:
        return STATUS_OCR_FAILED
    if STATUS_AUTHORIZED in statuses:
        return STATUS_AUTHORIZED
    if STATUS_UNCERTAIN in statuses:
        return STATUS_UNCERTAIN
    if STATUS_UNAUTHORIZED in statuses:
        return STATUS_UNAUTHORIZED
    return STATUS_OCR_FAILED


def process_batch(directory: str | Path, variant: str = DEFAULT_MODEL_VARIANT) -> list[PipelineResult]:
    """Process every supported file in `directory` (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    exts = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)

    all_results: list[PipelineResult] = []
    for f in files:
        all_results.extend(process_image(f, variant=variant))
    return all_results


def print_summary(results: Iterable[PipelineResult]) -> None:
    rs = list(results)
    total = len(rs)
    if total == 0:
        print("[summary] no frames processed")
        return
    counts = {
        STATUS_AUTHORIZED: 0,
        STATUS_UNAUTHORIZED: 0,
        STATUS_UNCERTAIN: 0,
        STATUS_OCR_FAILED: 0,
        STATUS_NO_PLATE: 0,
    }
    total_ms = 0.0
    failed = 0
    for r in rs:
        counts[r.status] = counts.get(r.status, 0) + 1
        total_ms += r.processing_time_ms
        if r.error_category != ERR_SUCCESS:
            failed += 1
    print("─" * 60)
    print(f"[summary] frames processed: {total}")
    for s, n in counts.items():
        print(f"           {s:<16} {n}")
    print(f"           failed (non-SUCCESS category): {failed}")
    print(f"           avg processing time: {total_ms / total:.1f} ms")
    print("─" * 60)
