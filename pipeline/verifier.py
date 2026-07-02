"""
Stage 7 — Database Verifier.

Looks up the normalized plate in the SQLite authorized_plates table:
    1. Exact match (case-insensitive, stripped) → AUTHORIZED (EXACT)
    2. Otherwise Levenshtein distance against all active rows;
       distance ≤ 2 → AUTHORIZED (FUZZY) with the
       "FUZZY_MATCH - verify manually" flag.
    3. No match → UNAUTHORIZED

OCR confidence gating — two thresholds, deliberate spec refinement
------------------------------------------------------------------
The original spec specified one global threshold (0.70): any OCR
confidence below it forces UNCERTAIN regardless of match outcome.

Empirical testing on MIPA deployment images showed legitimate
authorized vehicles routinely producing OCR confidences in the
0.55–0.69 band (due to stencil font generalization, dim parking-
garage lighting, oblique angles) even when the OCR text matched a
database entry EXACTLY. The strict gate was routing the most-common
success case to UNCERTAIN, degrading usability without improving
safety — an EXACT DB match is *independent evidence* that the OCR
read is correct.

The refined logic:
  - EXACT matches use the relaxed threshold (default 0.50). When OCR
    confidence is in [0.50, 0.70) and the match is exact, status =
    AUTHORIZED with a flag indicating the relaxed gate fired (so the
    log makes the design choice transparent and auditable).
  - FUZZY matches and NO_MATCH cases keep the strict 0.70 — these are
    the cases where a false positive is most damaging, so the safety
    floor is preserved.

Document this in the thesis under "Design Refinement Based on
Empirical Evaluation."
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Optional

import Levenshtein

from config import (
    DB_PATH,
    FUZZY_FLAG,
    FUZZY_MAX_DIST,
    LOW_CONF_EXACT_FLAG,
    OCR_CONF_THRESH,
    OCR_CONF_THRESH_EXACT,
)
from pipeline import (
    STATUS_AUTHORIZED,
    STATUS_UNAUTHORIZED,
    STATUS_UNCERTAIN,
    VerificationResult,
)


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def verify(
    normalized_plate: str,
    ocr_confidence: float,
    detection_confidence: float,
) -> VerificationResult:
    """Stage 7 entry point."""
    if not normalized_plate:
        return VerificationResult(
            status=STATUS_UNCERTAIN,
            ocr_confidence=ocr_confidence,
            detection_confidence=detection_confidence,
        )

    needle = normalized_plate.strip().upper()
    # Two gates: strict default + relaxed for EXACT matches
    fails_strict_gate = ocr_confidence < OCR_CONF_THRESH
    fails_exact_gate = ocr_confidence < OCR_CONF_THRESH_EXACT

    with _conn() as conn:
        # 1. Exact match
        row = conn.execute(
            "SELECT plate_number FROM authorized_plates "
            "WHERE is_active = 1 AND UPPER(plate_number) = ? LIMIT 1",
            (needle,),
        ).fetchone()
        if row is not None:
            # Three sub-cases on the EXACT branch:
            #  a) conf >= 0.70 → AUTHORIZED (no flag)
            #  b) 0.50 ≤ conf < 0.70 → AUTHORIZED (relaxed-gate flag)
            #  c) conf < 0.50 → UNCERTAIN (even the relaxed gate failed)
            if fails_exact_gate:
                status = STATUS_UNCERTAIN
                flag = "LOW_OCR_CONFIDENCE - verify manually"
            elif fails_strict_gate:
                status = STATUS_AUTHORIZED
                flag = LOW_CONF_EXACT_FLAG
            else:
                status = STATUS_AUTHORIZED
                flag = None
            return VerificationResult(
                status=status,
                matched_plate=row["plate_number"],
                match_type="EXACT",
                ocr_confidence=ocr_confidence,
                detection_confidence=detection_confidence,
                flag=flag,
            )

        # 2. Fuzzy — keeps the strict 0.70 gate
        rows = conn.execute(
            "SELECT plate_number FROM authorized_plates WHERE is_active = 1"
        ).fetchall()
        best_plate: Optional[str] = None
        best_dist = 10**9
        for r in rows:
            d = Levenshtein.distance(needle, r["plate_number"].upper())
            if d < best_dist:
                best_dist = d
                best_plate = r["plate_number"]

        if best_plate is not None and best_dist <= FUZZY_MAX_DIST:
            return VerificationResult(
                status=STATUS_UNCERTAIN if fails_strict_gate else STATUS_AUTHORIZED,
                matched_plate=best_plate,
                match_type="FUZZY",
                ocr_confidence=ocr_confidence,
                detection_confidence=detection_confidence,
                flag=(
                    "LOW_OCR_CONFIDENCE - verify manually"
                    if fails_strict_gate
                    else FUZZY_FLAG
                ),
            )

    # 3. No match — strict gate too
    return VerificationResult(
        status=STATUS_UNCERTAIN if fails_strict_gate else STATUS_UNAUTHORIZED,
        match_type="NONE",
        ocr_confidence=ocr_confidence,
        detection_confidence=detection_confidence,
        flag=("LOW_OCR_CONFIDENCE - verify manually" if fails_strict_gate else None),
    )
