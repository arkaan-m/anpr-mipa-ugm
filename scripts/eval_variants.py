"""
Post-Colab variant comparison finalizer.

After running notebook 07_variant_comparison.ipynb on Colab and downloading:
  - models/best_yolov8n.pt
  - models/best_yolov8m.pt
  - data/variant_comparison.csv   (from Colab)

Run this script to print a clean thesis-ready table and save the canonical
results to data/variant_comparison_final.csv.

Usage:
    venv/bin/python scripts/eval_variants.py

If you want to re-run the MIPA test-set eval locally (slow on CPU):
    venv/bin/python scripts/eval_variants.py --reeval
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VARIANTS = [
    ("YOLOv8n", REPO_ROOT / "models" / "best_yolov8n.pt"),
    ("YOLOv8s", REPO_ROOT / "models" / "best_yolov8s.pt"),
    ("YOLOv8m", REPO_ROOT / "models" / "best_yolov8m.pt"),
]

COLAB_CSV = REPO_ROOT / "data" / "variant_comparison.csv"
FINAL_CSV = REPO_ROOT / "data" / "variant_comparison_final.csv"
DATA_YAML  = REPO_ROOT / "data" / "processed" / "mipa_v1" / "data.yaml"
TEST_IMGS  = REPO_ROOT / "data" / "processed" / "mipa_v1" / "test" / "images"


def _col(s, w):
    return str(s).ljust(w)


def _print_table(rows: list[dict]):
    print("\n" + "=" * 74)
    print("  YOLOV8 VARIANT COMPARISON — MIPA TEST SET")
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

    # recommendation
    DET_BUDGET_MS = 1500
    candidates = [r for r in rows if float(r["cpu_ms"]) < DET_BUDGET_MS]
    pool = candidates if candidates else rows
    winner = max(pool, key=lambda r: float(r["map50"]))
    print(f"\n  Best variant (mAP50 under {DET_BUDGET_MS}ms det budget): {winner['variant']}")
    print(f"  mAP50={float(winner['map50']):.4f}, CPU={float(winner['cpu_ms']):.0f}ms/img")


def _reeval() -> list[dict]:
    """Re-run MIPA test-set eval locally. Slow on CPU."""
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"[error] {DATA_YAML} not found")
        sys.exit(1)

    test_images = sorted(
        p for p in TEST_IMGS.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows = []
    for name, weights in VARIANTS:
        if not weights.exists():
            print(f"[skip] {weights.name} not found")
            continue
        print(f"\n--- {name} ---")
        m = YOLO(str(weights))

        metrics = m.val(data=str(DATA_YAML), split="test", imgsz=640,
                        device="cpu", verbose=False)
        map50   = metrics.box.map50
        map5095 = metrics.box.map
        prec    = metrics.box.mp
        rec     = metrics.box.mr

        m.predict(str(test_images[0]), imgsz=640, device="cpu", verbose=False)
        t0 = time.perf_counter()
        for p in test_images:
            m.predict(str(p), imgsz=640, device="cpu", verbose=False)
        avg_ms = (time.perf_counter() - t0) / len(test_images) * 1000

        rows.append({
            "variant": name, "map50": map50, "map5095": map5095,
            "precision": prec, "recall": rec, "cpu_ms": avg_ms,
        })
        print(f"  mAP50={map50:.4f}  CPU={avg_ms:.0f}ms/img")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reeval", action="store_true",
                        help="Re-run MIPA test eval locally instead of reading Colab CSV")
    args = parser.parse_args()

    if args.reeval:
        print("[mode] re-evaluating locally (this will take a while on CPU)...")
        rows = _reeval()
    else:
        if not COLAB_CSV.exists():
            print(f"[error] {COLAB_CSV} not found.\n"
                  "  Download variant_comparison.csv from Colab Drive and place it in data/\n"
                  "  Or run with --reeval to evaluate locally.")
            sys.exit(1)
        with open(COLAB_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"[load] {COLAB_CSV.name}  ({len(rows)} variants)")

    _print_table(rows)

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant","map50","map5095","precision","recall","cpu_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] → {FINAL_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
