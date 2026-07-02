"""
Evaluate OCR accuracy — EasyOCR vs Tesseract vs combined pipeline.

Metrics
-------
CLA  Character-Level Accuracy   avg(max(0, len(gt) - edit_dist(pred, gt)) / len(gt))
PLA  Plate-Level Accuracy        exact matches / total plates evaluated

Usage
-----
    venv/bin/python scripts/eval_ocr.py --gt data/ocr_gt.csv

Ground truth CSV (create this manually by reading each plate image):
    image,plate
    IMG_5706.jpg,AB 1234 CD
    IMG_5740.jpg,B 5678 F

Notes
-----
- Images are looked up in --images-dir (default: data/raw/mipa_photos/).
- The script runs the full pipeline: detection → crop → both OCR engines.
- Images where detection fails are listed separately and excluded from CLA/PLA.
- Ground truth is normalized the same way the pipeline normalizes (so "AB1234CD"
  and "AB 1234 CD" are treated identically).
- Results are saved to data/ocr_eval_results.csv for thesis tables.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _cla(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    dist = _edit_distance(pred, gt)
    correct = max(0, len(gt) - dist)
    return correct / len(gt)


def _normalize_gt(text: str) -> str:
    """Normalize ground truth the same way text_normalizer does."""
    try:
        from pipeline.text_normalizer import normalize
        return normalize(text)
    except Exception:
        return text.upper().strip()


def _load_gt(csv_path: Path, images_dir: Path) -> list[tuple[Path, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image", "").strip()
            plate_gt = row.get("plate", "").strip()
            if not img_name or not plate_gt:
                continue
            img_path = images_dir / img_name
            if not img_path.exists():
                print(f"  [warn] not found: {img_path.name}")
                continue
            rows.append((img_path, _normalize_gt(plate_gt)))
    return rows


def _run_pipeline(img_path: Path, model):
    """Run detection + crop + dual OCR. Returns (easy, tess, chosen) OcrOutputs or None on failure."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None, "cannot read image"

    from pipeline.preprocessor import preprocess
    from pipeline.detector import detect
    from pipeline.plate_extractor import extract
    from pipeline import ocr_engine

    pre = preprocess(bgr)
    boxes = detect(pre, model=model)
    if not boxes:
        return None, "no plate detected"

    best_box = max(boxes, key=lambda b: b.area)
    crop = extract(pre.original_bgr, best_box, img_path.name)
    chosen, easy, tess = ocr_engine.run(crop.deskewed_bgr, crop.processed_gray)
    return (easy, tess, chosen), None


def _print_row(cols: list, widths: list[int]):
    print("  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy (EasyOCR vs Tesseract)")
    parser.add_argument("--gt", required=True, type=Path, help="Ground truth CSV (image,plate)")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "mipa_photos",
        help="Directory containing the images named in --gt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "ocr_eval_results.csv",
        help="Where to save per-image results",
    )
    args = parser.parse_args()

    if not args.gt.exists():
        print(f"[error] ground truth file not found: {args.gt}")
        sys.exit(1)

    print(f"\n[load] ground truth: {args.gt}")
    rows = _load_gt(args.gt, args.images_dir)
    if not rows:
        print("[error] no valid rows in ground truth CSV — check filenames and format")
        sys.exit(1)
    print(f"[load] {len(rows)} plates to evaluate")

    from pipeline.detector import load_model
    print("[load] loading YOLOv8 model ...")
    model = load_model()
    print("[load] model ready\n")

    results = []
    det_failures = []

    for img_path, gt_text in rows:
        print(f"  → {img_path.name} (gt: {gt_text})")
        ocr_outputs, err = _run_pipeline(img_path, model)
        if ocr_outputs is None:
            print(f"     SKIP — {err}")
            det_failures.append((img_path.name, gt_text, err))
            continue

        easy, tess, chosen = ocr_outputs
        results.append({
            "image": img_path.name,
            "gt": gt_text,
            "easy_text": easy.text,
            "easy_conf": easy.confidence,
            "easy_valid": easy.is_valid,
            "easy_cla": _cla(easy.text, gt_text),
            "easy_pla": int(easy.text == gt_text),
            "tess_text": tess.text,
            "tess_conf": tess.confidence,
            "tess_valid": tess.is_valid,
            "tess_cla": _cla(tess.text, gt_text),
            "tess_pla": int(tess.text == gt_text),
            "chosen_engine": chosen.engine,
            "chosen_text": chosen.text,
            "chosen_cla": _cla(chosen.text, gt_text),
            "chosen_pla": int(chosen.text == gt_text),
        })

    n = len(results)
    if n == 0:
        print("\n[error] all images failed detection — cannot compute metrics")
        sys.exit(1)

    easy_cla  = sum(r["easy_cla"]   for r in results) / n * 100
    tess_cla  = sum(r["tess_cla"]   for r in results) / n * 100
    comb_cla  = sum(r["chosen_cla"] for r in results) / n * 100
    easy_pla  = sum(r["easy_pla"]   for r in results) / n * 100
    tess_pla  = sum(r["tess_pla"]   for r in results) / n * 100
    comb_pla  = sum(r["chosen_pla"] for r in results) / n * 100

    print("\n" + "=" * 62)
    print("  OCR EVALUATION SUMMARY")
    print("=" * 62)
    w = [20, 10, 10]
    _print_row(["Engine", "CLA (%)", "PLA (%)"], w)
    print("  " + "-" * 44)
    _print_row(["EasyOCR",           f"{easy_cla:.1f}", f"{easy_pla:.1f}"], w)
    _print_row(["Tesseract",         f"{tess_cla:.1f}", f"{tess_pla:.1f}"], w)
    _print_row(["Combined pipeline", f"{comb_cla:.1f}", f"{comb_pla:.1f}"], w)
    print("=" * 62)
    print(f"  Evaluated: {n} plates  |  Detection failures: {len(det_failures)}")

    target_cla, target_pla = 85.0, 80.0
    for label, cla, pla in [
        ("EasyOCR",  easy_cla, easy_pla),
        ("Tesseract", tess_cla, tess_pla),
        ("Combined",  comb_cla, comb_pla),
    ]:
        c = "✓" if cla >= target_cla else "✗"
        p = "✓" if pla >= target_pla else "✗"
        print(f"  {label:20s}  CLA {c} (target ≥{target_cla:.0f}%)   PLA {p} (target ≥{target_pla:.0f}%)")
    print("=" * 62)

    print("\n  Per-image breakdown:")
    cols = ["Image", "GT", "Easy", "Tess", "Chosen", "E✓", "T✓", "C✓"]
    cw   = [20, 12, 12, 12, 12, 3, 3, 3]
    _print_row(cols, cw)
    print("  " + "-" * (sum(cw) + 2 * len(cw)))
    for r in results:
        _print_row([
            r["image"][:19],
            r["gt"],
            r["easy_text"][:11],
            r["tess_text"][:11],
            r["chosen_text"][:11],
            "✓" if r["easy_pla"]   else "✗",
            "✓" if r["tess_pla"]   else "✗",
            "✓" if r["chosen_pla"] else "✗",
        ], cw)

    if det_failures:
        print(f"\n  Detection failures ({len(det_failures)}):")
        for name, gt, reason in det_failures:
            print(f"    {name}: {reason} (gt: {gt})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[save] per-image results → {args.out.relative_to(REPO_ROOT)}")

    print("\n  Thesis table values:")
    print(f"  EasyOCR   CLA={easy_cla:.1f}%  PLA={easy_pla:.1f}%")
    print(f"  Tesseract CLA={tess_cla:.1f}%  PLA={tess_pla:.1f}%")
    print(f"  Combined  CLA={comb_cla:.1f}%  PLA={comb_pla:.1f}%\n")


if __name__ == "__main__":
    main()
