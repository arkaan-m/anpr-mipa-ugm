"""
Stage 6 — Text Normalizer.

Uppercases, strips non-alphanumeric chars (except space), collapses
whitespace to exactly one space between the three plate segments
(region | digits | suffix), and validates the region code against the
authoritative Indonesian region list from config.yaml.

Raises ValidationError if the first 1–2 chars don't match a known region.
"""

from __future__ import annotations

import re

from config import REGION_CODES
from pipeline import ValidationError


_ALNUM_OR_SPACE = re.compile(r"[^A-Z0-9 ]+")
_WS = re.compile(r"\s+")
# Best-effort 3-token split: (alpha 1–2) (digit 1–4) (alpha 0–3, optional)
_SEGMENT_RE = re.compile(r"^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{0,3})$")


def normalize(text: str) -> str:
    """Return canonical 'AB 1234 CD' form. Raises ValidationError on bad input."""
    if not text:
        raise ValidationError("empty text")

    upper = text.upper()
    cleaned = _ALNUM_OR_SPACE.sub("", upper)
    collapsed = _WS.sub(" ", cleaned).strip()
    if not collapsed:
        raise ValidationError(f"no alphanumerics after cleaning: '{text}'")

    # Strip spaces entirely, then re-split using the segment regex (handles
    # both "AB1234CD" and "AB 1234 CD" and " A B 1 2 3 4 ").
    no_space = collapsed.replace(" ", "")
    m = _SEGMENT_RE.match(no_space)
    if m is None:
        raise ValidationError(f"does not match Indonesian plate structure: '{collapsed}'")

    region, digits, suffix = m.group(1), m.group(2), m.group(3)
    if region not in REGION_CODES:
        raise ValidationError(f"unknown region code '{region}' (from '{collapsed}')")

    return f"{region} {digits} {suffix}".strip() if suffix else f"{region} {digits}"
