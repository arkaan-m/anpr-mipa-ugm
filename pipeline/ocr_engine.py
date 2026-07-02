"""
Stage 5 — Dual OCR Engine.

Runs EasyOCR and Tesseract in parallel on the processed plate image,
validates each against Indonesian plate regexes, and selects the best
output using:
    1. Both valid → higher confidence wins
    2. Only one valid → that one
    3. Neither valid → apply char substitutions, re-validate
    4. Still neither valid → OCR_FAILED

Both engines now apply a *text-height filter*: only detections whose
bounding-box height is ≥ MIN_TEXT_HEIGHT_RATIO of the tallest detection
are kept. This drops dealer-ad text, year stickers, and other small
text under the main plate row.

Returns: (chosen OcrOutput, easyocr OcrOutput, tesseract OcrOutput).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import (
    CHAR_SUBS,
    EASYOCR_GPU,
    EASYOCR_LANGS,
    MIN_TEXT_HEIGHT_RATIO,
    PLATE_REGEX_CAR,
    PLATE_REGEX_MOTO,
    TESSERACT_CONFIG,
)
from pipeline import OcrOutput, STATUS_OCR_FAILED


_RE_CAR = re.compile(PLATE_REGEX_CAR)
_RE_MOTO = re.compile(PLATE_REGEX_MOTO)
_NON_PLATE = re.compile(r"[^A-Z0-9 ]+")
_MULTI_WS = re.compile(r"\s+")

_EASY_READER = None


def _easyocr_reader():
    global _EASY_READER
    if _EASY_READER is None:
        import easyocr
        _EASY_READER = easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU, verbose=False)
    return _EASY_READER


def _clean(text: str) -> str:
    """Strip non-alphanumeric noise (asterisks, dots, dashes, bolts read as
    punctuation, etc.) before regex validation. Plates don't contain
    punctuation, so anything that isn't [A-Z0-9 ] is noise."""
    if not text:
        return text
    return _MULTI_WS.sub(" ", _NON_PLATE.sub("", text.upper())).strip()


def is_valid_plate(text: str) -> bool:
    t = _clean(text)
    return bool(_RE_CAR.match(t) or _RE_MOTO.match(t))


# ───────────────────────── Height filter helpers ────────────────────────

def _easyocr_bbox_height(bbox) -> float:
    """EasyOCR bbox is a 4-point polygon: [[x,y], [x,y], [x,y], [x,y]]."""
    ys = [float(p[1]) for p in bbox]
    return max(ys) - min(ys)


def _filter_by_height(items: list, heights: list[float]) -> list[int]:
    """Return indices of items whose height ≥ ratio * max_height.
    Safety net: if filter drops everything, keep all items."""
    if not heights:
        return list(range(len(items)))
    max_h = max(heights)
    if max_h <= 0:
        return list(range(len(items)))
    cutoff = max_h * MIN_TEXT_HEIGHT_RATIO
    keep = [i for i, h in enumerate(heights) if h >= cutoff]
    return keep if keep else list(range(len(items)))


# ─────────────────────────── EasyOCR ────────────────────────────────────

_MIN_OCR_HEIGHT = 200  # upscale crops shorter than this before EasyOCR


def _upscale_for_ocr(img: np.ndarray) -> np.ndarray:
    """EasyOCR confidence drops sharply on small crops. Upscale short crops
    to a minimum height with cubic interpolation."""
    import cv2
    h = img.shape[0]
    if h >= _MIN_OCR_HEIGHT:
        return img
    scale = _MIN_OCR_HEIGHT / h
    new_w = int(round(img.shape[1] * scale))
    return cv2.resize(img, (new_w, _MIN_OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)


def _try_easyocr(plate_img: np.ndarray) -> OcrOutput:
    plate_img = _upscale_for_ocr(plate_img)
    try:
        reader = _easyocr_reader()
        results = reader.readtext(plate_img, detail=1, paragraph=False)
    except Exception:
        return OcrOutput(text="", confidence=0.0, engine="easyocr", is_valid=False)

    if not results:
        return OcrOutput(text="", confidence=0.0, engine="easyocr", is_valid=False)

    heights = [_easyocr_bbox_height(r[0]) for r in results]
    keep_idx = _filter_by_height(results, heights)
    filtered = [results[i] for i in keep_idx]

    # Left→right by leftmost-x of bbox
    filtered.sort(key=lambda r: min(p[0] for p in r[0]))
    pieces = [r[1] for r in filtered]
    confs = [float(r[2]) for r in filtered]
    text = _clean(" ".join(pieces))
    conf = float(np.mean(confs)) if confs else 0.0
    return OcrOutput(text=text, confidence=conf, engine="easyocr", is_valid=is_valid_plate(text))


# ─────────────────────────── Tesseract ──────────────────────────────────

def _try_tesseract(plate_img: np.ndarray) -> OcrOutput:
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            plate_img, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return OcrOutput(text="", confidence=0.0, engine="tesseract", is_valid=False)

    items: list[tuple[str, float, float, float]] = []  # (word, conf, left, height)
    for txt, c, left, h in zip(
        data.get("text", []), data.get("conf", []),
        data.get("left", []), data.get("height", []),
    ):
        s = (txt or "").strip()
        if not s:
            continue
        try:
            ci = float(c)
            hh = float(h)
            lx = float(left)
        except (TypeError, ValueError):
            continue
        if ci < 0:
            continue
        items.append((s, ci / 100.0, lx, hh))

    if not items:
        return OcrOutput(text="", confidence=0.0, engine="tesseract", is_valid=False)

    heights = [it[3] for it in items]
    keep_idx = _filter_by_height(items, heights)
    filtered = [items[i] for i in keep_idx]
    filtered.sort(key=lambda t: t[2])  # left→right

    words = [t[0] for t in filtered]
    confs = [t[1] for t in filtered]
    text = _clean(" ".join(words))
    conf = float(np.mean(confs)) if confs else 0.0
    return OcrOutput(text=text, confidence=conf, engine="tesseract", is_valid=is_valid_plate(text))


# ─────────────────────────── Substitutions ──────────────────────────────

def _is_fully_valid(text: str) -> bool:
    """Plate regex match AND valid region code."""
    if not is_valid_plate(text):
        return False
    try:
        from pipeline.text_normalizer import normalize
        normalize(text)
        return True
    except Exception:
        return False


def _apply_substitutions(text: str) -> str:
    """Try to coerce an invalid plate string into a valid one by swapping
    visually-similar characters. Each char may map to multiple candidates
    (e.g. L → [4, 1]). We enumerate 1- and 2-substitution variants in a
    *deterministic* order (config-file order is the preference order) and
    return the first variant that passes both regex and region-code validation.
    """
    if not text:
        return text

    seen: set[str] = {text}
    candidates: list[str] = [text]

    # 1-substitution variants — preserve config order
    for i, ch in enumerate(text):
        for sub in CHAR_SUBS.get(ch, ()):
            v = text[:i] + sub + text[i + 1 :]
            if v not in seen:
                seen.add(v)
                candidates.append(v)

    # 2-substitution variants — bounded, still order-preserving
    for v in list(candidates):
        for i, ch in enumerate(v):
            for sub in CHAR_SUBS.get(ch, ()):
                w = v[:i] + sub + v[i + 1 :]
                if w not in seen:
                    seen.add(w)
                    candidates.append(w)

    for c in candidates:
        if _is_fully_valid(c):
            return c

    # Positional fallback: Indonesian plates are always
    # region(1-2 letters) + digits(1-4) + suffix(0-3 letters). If the blind
    # swaps failed, force each segment to its correct character class.
    positional = _positional_fix(text)
    if positional and _is_fully_valid(positional):
        return positional
    return text


# Directional maps: a letter misread in a digit slot → digit, and vice-versa.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
             "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
              "6": "G", "8": "B"}


def _positional_fix(text: str) -> str:
    """Coerce a raw OCR string into the canonical region/digits/suffix
    structure by forcing each segment to its known character class. Tries all
    plausible segment boundaries and returns the first that yields a valid
    region code."""
    raw = "".join(ch for ch in text.upper() if ch.isalnum())
    n = len(raw)
    if n < 4:
        return ""

    best = ""
    # region length 1-2, suffix length 0-3, middle (digits) ≥ 1
    for rlen in (2, 1):
        for slen in (3, 2, 1, 0):
            dlen = n - rlen - slen
            if dlen < 1 or dlen > 4:
                continue
            region = "".join(_TO_LETTER.get(c, c) for c in raw[:rlen])
            digits = "".join(_TO_DIGIT.get(c, c) for c in raw[rlen:rlen + dlen])
            suffix = "".join(_TO_LETTER.get(c, c) for c in raw[rlen + dlen:])
            if not region.isalpha() or not digits.isdigit():
                continue
            if suffix and not suffix.isalpha():
                continue
            candidate = f"{region} {digits} {suffix}".strip()
            if _is_fully_valid(candidate):
                return candidate
    return best


# ────────────────────────────── Run ─────────────────────────────────────

def run(color_img: np.ndarray, gray_img: np.ndarray) -> tuple[OcrOutput, OcrOutput, OcrOutput]:
    """Returns (chosen, easyocr, tesseract). `chosen.engine` may be OCR_FAILED.

    `color_img` should be the deskewed BGR/color crop — EasyOCR is a
    natural-scene recognizer and binarized input degrades its confidence.
    `gray_img` should be the binarized + denoised + sharpened crop — that's
    what Tesseract's LSTM model is happiest with.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_easy = pool.submit(_try_easyocr, color_img)
        f_tess = pool.submit(_try_tesseract, gray_img)
        easy = f_easy.result()
        tess = f_tess.result()

    chosen = _select(easy, tess)
    return chosen, easy, tess


def _select(easy: OcrOutput, tess: OcrOutput) -> OcrOutput:
    if easy.is_valid and tess.is_valid:
        return easy if easy.confidence >= tess.confidence else tess
    if easy.is_valid and not tess.is_valid:
        return easy
    if tess.is_valid and not easy.is_valid:
        return tess

    # Substitution rescue: if a 1- or 2-char swap produces a structurally
    # valid plate, that's a *more* confident outcome (we now know the
    # structure matches), not a less confident one. We pass the engine's
    # raw confidence through unmodified.
    easy_fixed = _apply_substitutions(easy.text)
    tess_fixed = _apply_substitutions(tess.text)
    easy_v = is_valid_plate(easy_fixed)
    tess_v = is_valid_plate(tess_fixed)

    if easy_v and tess_v:
        pe = OcrOutput(text=easy_fixed, confidence=easy.confidence, engine="easyocr+sub", is_valid=True)
        pt = OcrOutput(text=tess_fixed, confidence=tess.confidence, engine="tesseract+sub", is_valid=True)
        return pe if pe.confidence >= pt.confidence else pt
    if easy_v:
        return OcrOutput(text=easy_fixed, confidence=easy.confidence, engine="easyocr+sub", is_valid=True)
    if tess_v:
        return OcrOutput(text=tess_fixed, confidence=tess.confidence, engine="tesseract+sub", is_valid=True)

    best = easy if easy.confidence >= tess.confidence else tess
    return OcrOutput(
        text=best.text, confidence=best.confidence, engine=STATUS_OCR_FAILED, is_valid=False,
    )
