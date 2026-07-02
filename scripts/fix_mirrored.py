"""
Detect and fix horizontally mirrored images in the MIPA raw photo folder.

Selfie-mode (front-camera) photos appear mirrored. This script:
  1. Runs YOLOv8 to detect the license plate crop
  2. Tries EasyOCR on the normal crop, then the horizontally flipped crop
  3. Checks which orientation produces text matching the Indonesian plate regex
  4. If the flipped version is better, overwrites the image file and re-runs
     auto-labeling to regenerate the corresponding .txt label

Usage:
    venv/bin/python scripts/fix_mirrored.py --input data/raw/mipa_photos
    venv/bin/python scripts/fix_mirrored.py --input data/raw/mipa_photos --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Indonesian plate regex (both civilian and motorcycle)
PLATE_RE = re.compile(
    r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$'
)

CONF_THRESHOLD = 0.25  # lower than usual to catch more crops for testing


def _letterbox(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas


def _crop_plate(bgr: np.ndarray, model) -> np.ndarray | None:
    """Return the best plate crop from the image, or None."""
    lb = _letterbox(bgr, 1280)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, imgsz=1280, conf=CONF_THRESHOLD,
                            device='cpu', verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Pick highest-confidence box
    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
    # Scale coords back to original image
    h_orig, w_orig = bgr.shape[:2]
    scale = 1280 / max(h_orig, w_orig)
    pad_x = (1280 - int(w_orig * scale)) // 2
    pad_y = (1280 - int(h_orig * scale)) // 2
    ox1 = max(0, int((x1 - pad_x) / scale))
    oy1 = max(0, int((y1 - pad_y) / scale))
    ox2 = min(w_orig, int((x2 - pad_x) / scale))
    oy2 = min(h_orig, int((y2 - pad_y) / scale))
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    return bgr[oy1:oy2, ox1:ox2]


def _ocr_text(crop_bgr: np.ndarray, reader) -> str:
    """Run EasyOCR on a crop, return cleaned uppercase text."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=0)
    raw = ' '.join(results).upper()
    cleaned = re.sub(r'[^A-Z0-9 ]', '', raw).strip()
    # Collapse multiple spaces
    return re.sub(r'\s+', ' ', cleaned)


def _score(text: str) -> int:
    """0 = no match, 1 = partial (has digits+letters), 2 = full plate match."""
    if PLATE_RE.match(text):
        return 2
    if re.search(r'[A-Z]', text) and re.search(r'\d', text):
        return 1
    return 0


def _relabel(img_path: Path, model):
    """Regenerate the YOLO .txt label for a (now-fixed) image."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    lb = _letterbox(bgr, 1280)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, imgsz=1280, conf=0.30,
                            device='cpu', verbose=False)
    boxes = results[0].boxes
    txt_path = img_path.with_suffix('.txt')
    if boxes is None or len(boxes) == 0:
        txt_path.write_text('')
        return
    scale = 1280 / max(h, w)
    pad_x = (1280 - int(w * scale)) // 2
    pad_y = (1280 - int(h * scale)) // 2
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        ox1 = (x1 - pad_x) / scale / w
        oy1 = (y1 - pad_y) / scale / h
        ox2 = (x2 - pad_x) / scale / w
        oy2 = (y2 - pad_y) / scale / h
        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2
        bw = ox2 - ox1
        bh = oy2 - oy1
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default=REPO_ROOT / 'data' / 'raw' / 'mipa_photos')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report mirrored images without fixing them')
    args = parser.parse_args()

    images = sorted(args.input.glob('*.JPG')) + sorted(args.input.glob('*.jpg'))
    if not images:
        print(f'[error] no JPG images found in {args.input}')
        sys.exit(1)

    print(f'[info] {len(images)} images to check')
    print('[load] loading YOLOv8 model...')
    from ultralytics import YOLO
    model = YOLO(str(REPO_ROOT / 'models' / 'best_yolov8s.pt'))

    print('[load] loading EasyOCR...')
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    print('[scan] checking for mirrored images...\n')

    mirrored = []
    no_plate = []

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        crop = _crop_plate(bgr, model)
        if crop is None:
            no_plate.append(img_path.name)
            continue

        text_normal = _ocr_text(crop, reader)
        text_flipped = _ocr_text(cv2.flip(crop, 1), reader)

        score_n = _score(text_normal)
        score_f = _score(text_flipped)

        if score_f > score_n:
            mirrored.append(img_path.name)
            status = f'MIRRORED  normal="{text_normal}"({score_n})  flipped="{text_flipped}"({score_f})'
        else:
            status = f'ok        "{text_normal}"'

        print(f'  {img_path.name:25s}  {status}')

    print(f'\n[result] {len(mirrored)} mirrored, {len(no_plate)} no-plate, '
          f'{len(images) - len(mirrored) - len(no_plate)} normal')

    if mirrored and not args.dry_run:
        print('\n[fix] flipping mirrored images...')
        for name in mirrored:
            img_path = args.input / name
            bgr = cv2.imread(str(img_path))
            flipped = cv2.flip(bgr, 1)
            cv2.imwrite(str(img_path), flipped)
            _relabel(img_path, model)
            print(f'  flipped + relabeled: {name}')
        print(f'[done] fixed {len(mirrored)} images')
    elif args.dry_run and mirrored:
        print('\n[dry-run] no files changed. Re-run without --dry-run to fix.')


if __name__ == '__main__':
    main()
