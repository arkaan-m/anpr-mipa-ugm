"""
YOLOv5s pretrained baseline evaluation on the MIPA test set.

The proposal (Section 2.2 / Table 4.4) includes YOLOv5s as a reference
comparator in the variant comparison table — specifically to show how much
the YOLOv8 finetune gained over the prior-generation architecture.

This script runs the stock pretrained yolov5s.pt (no MIPA finetuning) on
the MIPA test set and appends the result to variant_comparison_final.csv
so the thesis table has all four rows: YOLOv5s | YOLOv8n | YOLOv8s | YOLOv8m.

Usage:
    venv/bin/python scripts/eval_yolov5_baseline.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_YAML   = REPO_ROOT / "data" / "processed" / "mipa_v1" / "data.yaml"
TEST_IMGS   = REPO_ROOT / "data" / "processed" / "mipa_v1" / "test" / "images"
FINAL_CSV   = REPO_ROOT / "data" / "variant_comparison_final.csv"


def _col(s, w):
    return str(s).ljust(w)


def _print_table(rows: list[dict]):
    print("\n" + "=" * 74)
    print("  YOLO VARIANT COMPARISON — MIPA TEST SET (incl. YOLOv5s baseline)")
    print("=" * 74)
    widths = [12, 10, 14, 11, 9, 12]
    headers = ["Variant", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "CPU ms/img"]
    print("  " + "  ".join(_col(h, widths[i]) for i, h in enumerate(headers)))
    print("  " + "-" * 70)
    for r in rows:
        cols = [
            r["variant"],
            f"{float(r['map50']):.4f}",
            f"{float(r['map5095']):.4f}",
            f"{float(r['precision']):.4f}",
            f"{float(r['recall']):.4f}",
            f"{float(r['cpu_ms']):.0f}",
        ]
        print("  " + "  ".join(_col(v, widths[i]) for i, v in enumerate(cols)))
    print("=" * 74)


def main():
    if not DATA_YAML.exists():
        print(f"[error] {DATA_YAML} not found")
        sys.exit(1)

    test_images = sorted(TEST_IMGS.glob("*"))
    if not test_images:
        print(f"[error] no images in {TEST_IMGS}")
        sys.exit(1)
    print(f"[info] {len(test_images)} test images")

    # Load existing results
    existing_rows: list[dict] = []
    if FINAL_CSV.exists():
        with open(FINAL_CSV, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        # Remove any prior yolov5s row so we don't duplicate
        existing_rows = [r for r in existing_rows if "yolov5" not in r["variant"].lower()]

    from ultralytics import YOLO

    print("\n[load] downloading / loading pretrained yolov5s.pt ...")
    try:
        model = YOLO("yolov5s.pt")
    except Exception as e:
        print(f"[error] could not load yolov5s.pt via ultralytics: {e}")
        print("  Try: venv/bin/pip install ultralytics --upgrade")
        sys.exit(1)

    print("[eval] running on MIPA test set (CPU) ...")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        device="cpu",
        verbose=False,
    )
    map50   = metrics.box.map50
    map5095 = metrics.box.map
    prec    = metrics.box.mp
    rec     = metrics.box.mr

    # CPU inference time
    model.predict(str(test_images[0]), imgsz=640, device="cpu", verbose=False)  # warmup
    t0 = time.perf_counter()
    for p in test_images:
        model.predict(str(p), imgsz=640, device="cpu", verbose=False)
    avg_ms = (time.perf_counter() - t0) / len(test_images) * 1000

    yolov5_row = {
        "variant":   "YOLOv5s*",
        "map50":     map50,
        "map5095":   map5095,
        "precision": prec,
        "recall":    rec,
        "cpu_ms":    avg_ms,
    }
    print(f"\n  YOLOv5s*  mAP50={map50:.4f}  CPU={avg_ms:.0f}ms/img")
    print("  (* pretrained only — no MIPA finetuning)")

    # Build final table: YOLOv5s first (reference), then n/s/m
    all_rows = [yolov5_row] + existing_rows

    _print_table(all_rows)
    print("\n  * YOLOv5s is pretrained only (no MIPA finetuning) — reference baseline")
    print("    YOLOv8 variants are finetuned on 54 MIPA training images")

    # Save
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "map50", "map5095", "precision", "recall", "cpu_ms"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[save] → {FINAL_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
