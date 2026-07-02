"""
Preprocessing ablation study — 4 conditions (proposal Section 4.7.5 Experiment B).

All conditions use 640×640 input (the deployed resolution).
Deskewing and binarization are always active in plate_extractor, so
Condition 4 is identical to Condition 3 in this implementation.

  1  No preprocessing        — 1280×1280, no CLAHE, no Wiener
  2  CLAHE only              — 1280×1280 + CLAHE
  3  CLAHE + Deblurring      — 1280×1280 + CLAHE + Wiener
  4  Full pipeline           — same as 3 (deskew/binarize always on in plate_extractor)

Metrics per condition:
  Det rate   plates detected / total images
  CLA        character-level accuracy (EasyOCR chosen output vs ground truth)
  PLA        plate-level accuracy (exact match rate)
  Sys acc    correct system decisions / total (using system_gt.csv)
  FAR        false accept rate on unauthorized plates
  FRR        false reject rate on authorized plates

Usage:
    venv/bin/python scripts/eval_ablation.py \\
        --ocr-gt  data/ocr_gt.csv \\
        --sys-gt  data/system_gt.csv \\
        --images-dir data/processed/mipa_v1/test/images
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers shared with eval_ocr / eval_system
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _cla(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    return max(0, len(gt) - _edit_distance(pred, gt)) / len(gt)


def _normalize(text: str) -> str:
    try:
        from pipeline.text_normalizer import normalize
        return normalize(text)
    except Exception:
        return text.upper().strip()


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "label": "1 — No preprocessing (640)",
        "input_size": (640, 640),
        "enable_clahe": False,
        "enable_wiener": False,
    },
    {
        "label": "2 — CLAHE only (640)",
        "input_size": (640, 640),
        "enable_clahe": True,
        "enable_wiener": False,
    },
    {
        "label": "3 — CLAHE + Deblurring (640)",
        "input_size": (640, 640),
        "enable_clahe": True,
        "enable_wiener": True,
    },
    {
        "label": "4 — Full pipeline (640)*",
        "input_size": (640, 640),
        "enable_clahe": True,
        "enable_wiener": True,
    },
]


def _patch_preprocessor(cond: dict):
    import pipeline.preprocessor as pp
    pp.INPUT_SIZE = cond["input_size"]
    pp.ENABLE_CLAHE = cond["enable_clahe"]
    pp.ENABLE_WIENER = cond["enable_wiener"]


def _run_image(img_path: Path, model) -> dict[str, Any]:
    import cv2
    from pipeline.preprocessor import preprocess
    from pipeline.detector import detect
    from pipeline.plate_extractor import extract
    from pipeline import ocr_engine
    from pipeline.text_normalizer import normalize, ValidationError
    from pipeline.verifier import verify

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return {"detected": False, "error": "cannot read"}

    pre = preprocess(bgr)
    boxes = detect(pre, model=model)
    if not boxes:
        return {"detected": False, "error": "no detection"}

    best_box = max(boxes, key=lambda b: b.area)
    crop = extract(pre.original_bgr, best_box, img_path.name)
    chosen, easy, tess = ocr_engine.run(crop.deskewed_bgr, crop.processed_gray)

    try:
        normalized = normalize(chosen.text)
    except (ValidationError, Exception):
        return {"detected": True, "ocr_failed": True, "plate": chosen.text, "ocr_conf": chosen.confidence}

    result = verify(normalized, chosen.confidence, best_box.confidence)
    return {
        "detected": True,
        "ocr_failed": False,
        "plate": normalized,
        "ocr_conf": chosen.confidence,
        "status": result.status,
    }


# ---------------------------------------------------------------------------
# Evaluate one condition
# ---------------------------------------------------------------------------

def _eval_condition(cond: dict, ocr_rows: list[dict], sys_rows: list[dict], images_dir: Path, model) -> dict:
    _patch_preprocessor(cond)
    print(f"\n  [{cond['label']}]")

    # --- OCR metrics ---
    n_ocr = 0
    n_det_fail_ocr = 0
    cla_sum = 0.0
    pla_sum = 0

    for row in ocr_rows:
        img_path = images_dir / row["image"]
        if not img_path.exists():
            continue
        gt = _normalize(row["plate"])
        r = _run_image(img_path, model)
        if not r["detected"]:
            n_det_fail_ocr += 1
            print(f"    [det fail] {row['image']}")
            continue
        n_ocr += 1
        pred = r["plate"] if not r.get("ocr_failed") else r.get("plate", "")
        cla_sum += _cla(pred, gt)
        pla_sum += int(pred == gt)
        print(f"    {row['image']}  gt={gt}  pred={pred}  {'✓' if pred == gt else '✗'}")

    cla = cla_sum / n_ocr * 100 if n_ocr else 0.0
    pla = pla_sum / n_ocr * 100 if n_ocr else 0.0

    # --- System metrics ---
    n_sys = 0
    n_correct = 0
    n_auth = 0
    n_unauth = 0
    n_far = 0
    n_frr = 0
    n_det_fail_sys = 0

    for row in sys_rows:
        img_path = images_dir / row["image"]
        if not img_path.exists():
            continue
        n_sys += 1
        gt_auth = row["authorized"].strip().lower() in ("true", "1", "yes")
        if gt_auth:
            n_auth += 1
        else:
            n_unauth += 1

        r = _run_image(img_path, model)
        if not r["detected"] or r.get("ocr_failed"):
            status = "OCR_FAILED" if r.get("ocr_failed") else "NO_PLATE_FOUND"
            if r.get("detected") is False:
                n_det_fail_sys += 1
        else:
            status = r["status"]

        if status == "AUTHORIZED":
            correct = gt_auth
            is_far = not gt_auth
            is_frr = False
        elif status == "UNAUTHORIZED":
            correct = not gt_auth
            is_far = False
            is_frr = gt_auth
        elif status == "UNCERTAIN":
            correct = True
            is_far = False
            is_frr = False
        else:
            correct = False
            is_far = False
            is_frr = gt_auth

        n_correct += int(correct)
        n_far += int(is_far)
        n_frr += int(is_frr)

    acc = n_correct / n_sys * 100 if n_sys else 0.0
    far = n_far / n_unauth * 100 if n_unauth else 0.0
    frr = n_frr / n_auth * 100 if n_auth else 0.0
    det_rate_ocr = (len(ocr_rows) - n_det_fail_ocr) / len(ocr_rows) * 100 if ocr_rows else 0.0

    return {
        "label": cond["label"],
        "det_rate": det_rate_ocr,
        "n_detected": len(ocr_rows) - n_det_fail_ocr,
        "n_total_ocr": len(ocr_rows),
        "cla": cla,
        "pla": pla,
        "sys_acc": acc,
        "far": far,
        "frr": frr,
        "n_sys": n_sys,
        "n_det_fail_sys": n_det_fail_sys,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _col(s: str, w: int) -> str:
    return str(s).ljust(w)


def main():
    parser = argparse.ArgumentParser(description="Preprocessing ablation study")
    parser.add_argument("--ocr-gt",  type=Path, default=REPO_ROOT / "data" / "ocr_gt.csv")
    parser.add_argument("--sys-gt",  type=Path, default=REPO_ROOT / "data" / "system_gt.csv")
    parser.add_argument(
        "--images-dir", type=Path,
        default=REPO_ROOT / "data" / "processed" / "mipa_v1" / "test" / "images",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "ablation_results.csv")
    args = parser.parse_args()

    ocr_rows = _load_csv(args.ocr_gt) if args.ocr_gt.exists() else []
    sys_rows = _load_csv(args.sys_gt) if args.sys_gt.exists() else []
    if not ocr_rows and not sys_rows:
        print("[error] no ground truth files found")
        sys.exit(1)

    from pipeline.detector import load_model
    print("\n[load] loading YOLOv8 model ...")
    model = load_model()
    print("[load] ready")

    summary = []
    for cond in CONDITIONS:
        metrics = _eval_condition(cond, ocr_rows, sys_rows, args.images_dir, model)
        summary.append(metrics)

    # --- Print table ---
    print("\n\n" + "=" * 78)
    print("  PREPROCESSING ABLATION STUDY — RESULTS")
    print("=" * 78)
    w = [30, 10, 8, 8, 10, 8, 8]
    header = ["Condition", "Det rate", "CLA(%)", "PLA(%)", "Sys acc(%)", "FAR(%)", "FRR(%)"]
    print("  " + "  ".join(_col(h, w[i]) for i, h in enumerate(header)))
    print("  " + "-" * 74)
    for m in summary:
        row = [
            m["label"],
            f"{m['det_rate']:.1f}%",
            f"{m['cla']:.1f}",
            f"{m['pla']:.1f}",
            f"{m['sys_acc']:.1f}",
            f"{m['far']:.1f}",
            f"{m['frr']:.1f}",
        ]
        print("  " + "  ".join(_col(v, w[i]) for i, v in enumerate(row)))
    print("=" * 78)

    c1, c2, c3, c4 = summary[0], summary[1], summary[2], summary[3]

    print("\n  CLAHE contribution (Condition 1 → 2):")
    print(f"    Detection rate : {c1['det_rate']:.1f}% → {c2['det_rate']:.1f}%  (Δ{c2['det_rate']-c1['det_rate']:+.1f}%)")
    print(f"    CLA            : {c1['cla']:.1f}% → {c2['cla']:.1f}%  (Δ{c2['cla']-c1['cla']:+.1f}%)")
    print(f"    PLA            : {c1['pla']:.1f}% → {c2['pla']:.1f}%  (Δ{c2['pla']-c1['pla']:+.1f}%)")
    print(f"    System acc     : {c1['sys_acc']:.1f}% → {c2['sys_acc']:.1f}%  (Δ{c2['sys_acc']-c1['sys_acc']:+.1f}%)")

    print("\n  Wiener deblurring contribution (Condition 2 → 3):")
    print(f"    Detection rate : {c2['det_rate']:.1f}% → {c3['det_rate']:.1f}%  (Δ{c3['det_rate']-c2['det_rate']:+.1f}%)")
    print(f"    CLA            : {c2['cla']:.1f}% → {c3['cla']:.1f}%  (Δ{c3['cla']-c2['cla']:+.1f}%)")
    print(f"    PLA            : {c2['pla']:.1f}% → {c3['pla']:.1f}%  (Δ{c3['pla']-c2['pla']:+.1f}%)")

    print("\n  Overall (Condition 1 → 4, no preprocess → full pipeline):")
    print(f"    Detection rate : {c1['det_rate']:.1f}% → {c4['det_rate']:.1f}%  (Δ{c4['det_rate']-c1['det_rate']:+.1f}%)")
    print(f"    CLA            : {c1['cla']:.1f}% → {c4['cla']:.1f}%  (Δ{c4['cla']-c1['cla']:+.1f}%)")
    print(f"    System acc     : {c1['sys_acc']:.1f}% → {c4['sys_acc']:.1f}%  (Δ{c4['sys_acc']-c1['sys_acc']:+.1f}%)")
    print("\n  * Condition 4 = Condition 3: deskewing and binarization are always")
    print("    active in plate_extractor and cannot be toggled per condition.")
    print("=" * 78)

    # --- Save CSV ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[save] → {args.out.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
