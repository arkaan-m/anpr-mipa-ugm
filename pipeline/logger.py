"""
Stage 8 — Result Logger.

Two outputs per pipeline run:

  1. Append a row to results/logs/verification_log.csv with all spec
     fields (timestamp, filenames, OCR engine, confidences, status,
     error_category).
  2. Save an annotated copy of the original image to results/annotated/
     with a color-coded bounding box (GREEN=AUTHORIZED, RED=UNAUTHORIZED,
     YELLOW=UNCERTAIN, GRAY=OCR_FAILED/NO_PLATE) and an overlayed text
     badge showing the plate text + status.

Also mirrors each row into the SQLite `detection_logs` table for the
admin panel's transaction history.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    ANNOTATED_DIR,
    BBOX_COLORS,
    BBOX_THICKNESS,
    CSV_LOG_PATH,
    DB_PATH,
    FONT_SCALE,
    FONT_THICKNESS,
)
from pipeline import (
    ERR_DB_MISMATCH,
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
)


CSV_FIELDS = [
    "timestamp",
    "original_filename",
    "frame_index",
    "detected_plate_raw",
    "normalized_plate",
    "ocr_engine_used",
    "ocr_confidence",
    "detection_confidence",
    "verification_status",
    "match_type",
    "matched_plate",
    "error_category",
    "processing_time_ms",
    "annotated_path",
]


def log(result: PipelineResult) -> None:
    """Persist `result` to CSV + SQLite + annotated image (mutates `result.annotated_path`)."""
    annotated_path = _save_annotated(result)
    result.annotated_path = str(annotated_path) if annotated_path else None
    _append_csv(result)
    _append_sqlite(result)


# ───────────────────────────── CSV ─────────────────────────────

def _append_csv(result: PipelineResult) -> None:
    write_header = not CSV_LOG_PATH.exists()
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_for(result)
    with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _rows_for(result: PipelineResult) -> list[dict]:
    ts = datetime.now().isoformat(timespec="seconds")
    src = Path(result.source_path).name
    if not result.plates:
        return [{
            "timestamp": ts,
            "original_filename": src,
            "frame_index": result.frame_index,
            "detected_plate_raw": "",
            "normalized_plate": "",
            "ocr_engine_used": "",
            "ocr_confidence": 0.0,
            "detection_confidence": 0.0,
            "verification_status": result.status,
            "match_type": "NONE",
            "matched_plate": "",
            "error_category": result.error_category,
            "processing_time_ms": round(result.processing_time_ms, 1),
            "annotated_path": result.annotated_path or "",
        }]
    rows: list[dict] = []
    for p in result.plates:
        v = p.verification
        rows.append({
            "timestamp": ts,
            "original_filename": src,
            "frame_index": result.frame_index,
            "detected_plate_raw": p.raw_text,
            "normalized_plate": p.normalized_text,
            "ocr_engine_used": p.chosen_engine,
            "ocr_confidence": round(p.ocr_confidence, 3),
            "detection_confidence": round(p.bbox.confidence, 3),
            "verification_status": v.status if v else result.status,
            "match_type": v.match_type if v else "NONE",
            "matched_plate": (v.matched_plate or "") if v else "",
            "error_category": result.error_category,
            "processing_time_ms": round(result.processing_time_ms, 1),
            "annotated_path": result.annotated_path or "",
        })
    return rows


# ───────────────────────────── SQLite ──────────────────────────

def _append_sqlite(result: PipelineResult) -> None:
    rows = _rows_for(result)
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error:
        return  # don't crash pipeline because the log DB is missing
    try:
        for r in rows:
            conn.execute(
                """INSERT INTO detection_logs
                (original_filename, detected_plate_raw, normalized_plate,
                 ocr_engine_used, ocr_confidence, detection_confidence,
                 verification_status, match_type, matched_plate,
                 processing_time_ms, error_category, annotated_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r["original_filename"],
                    r["detected_plate_raw"],
                    r["normalized_plate"],
                    r["ocr_engine_used"],
                    r["ocr_confidence"],
                    r["detection_confidence"],
                    r["verification_status"],
                    r["match_type"],
                    r["matched_plate"],
                    r["processing_time_ms"],
                    r["error_category"],
                    r["annotated_path"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────── Annotation ────────────────────────

def _save_annotated(result: PipelineResult) -> Optional[Path]:
    try:
        img = cv2.imread(result.source_path, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        return None

    for p in result.plates:
        status = p.verification.status if p.verification else result.status
        color = BBOX_COLORS.get(status, (128, 128, 128))
        cv2.rectangle(img, (p.bbox.x1, p.bbox.y1), (p.bbox.x2, p.bbox.y2), color, BBOX_THICKNESS)
        label = _label_for(p, status)
        _draw_label(img, p.bbox.x1, max(0, p.bbox.y1 - 8), label, color)

    if not result.plates and result.status == STATUS_NO_PLATE:
        _draw_label(img, 10, 30, "NO PLATE DETECTED", BBOX_COLORS.get(STATUS_NO_PLATE, (128, 128, 128)))

    stem = Path(result.source_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ANNOTATED_DIR / f"{stem}_f{result.frame_index}_{ts}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def _label_for(p: PlateOutcome, status: str) -> str:
    plate = p.normalized_text or p.raw_text or "?"
    return f"{plate} | {status}"


def _draw_label(img: np.ndarray, x: int, y: int, text: str, color: tuple) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)
    pad = 4
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), color, thickness=-1)
    cv2.putText(
        img, text, (x + pad, y - pad),
        font, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, lineType=cv2.LINE_AA,
    )


def category_for(status: str) -> str:
    if status == STATUS_NO_PLATE:
        return ERR_DETECTION_FAILURE
    if status == STATUS_OCR_FAILED:
        return ERR_OCR_FAILURE
    if status == STATUS_UNAUTHORIZED:
        return ERR_DB_MISMATCH
    if status in (STATUS_AUTHORIZED, STATUS_UNCERTAIN):
        return ERR_SUCCESS
    return ERR_SUCCESS
