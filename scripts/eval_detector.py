"""
Evaluate a YOLOv8 detector against the test split of a prepared dataset.

Used in:
  - Phase 1 (baseline): score the existing models/best_yolov8s.pt on
    data/processed/public_v1/ to establish a "before" number
  - Phase 4 (after MIPA finetune): score the finetuned model on the same
    test split to quantify the improvement

Outputs:
  - mAP50, mAP50-95, precision, recall (overall)
  - per-image inference time (mean / median)
  - a one-page text summary saved alongside the model

Usage:
  venv/bin/python scripts/eval_detector.py
  venv/bin/python scripts/eval_detector.py --model models/best_yolov8s.pt \\
      --data data/processed/public_v1/data.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/best_yolov8s.pt", help="path to .pt weights")
    p.add_argument("--data", default="data/processed/public_v1/data.yaml", help="data.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="which split")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    # STANDARD mAP is computed at conf=0.001 (full precision-recall curve), which is
    # the ultralytics .val() default and the convention behind all published mAP numbers
    # (incl. the proposal's ≥85% target). A higher conf (e.g. 0.25) TRUNCATES the PR curve
    # and INFLATES the reported mAP — do not use it for the headline detection metric.
    p.add_argument("--conf", type=float, default=0.001, help="conf threshold for .val() mAP (0.001 = standard full-curve mAP; do NOT raise for headline numbers)")
    p.add_argument("--report", default=None, help="where to save the text summary (default: alongside the model)")
    args = p.parse_args()

    model_path = (REPO_ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    data_yaml = (REPO_ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    if not model_path.exists():
        print(f"[!] model not found: {model_path}", file=sys.stderr)
        return 2
    if not data_yaml.exists():
        print(f"[!] data.yaml not found: {data_yaml}", file=sys.stderr)
        return 2

    print(f"[load] model = {model_path}")
    print(f"[load] data  = {data_yaml}")
    print(f"[load] split = {args.split}")
    print(f"[load] imgsz = {args.imgsz}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    print(f"\n[eval] running .val() ...")
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
        plots=False,
    )

    # Measure per-image inference time on the same split (separate from .val)
    print(f"\n[time] measuring per-image inference time ...")
    import yaml
    with data_yaml.open() as f:
        dcfg = yaml.safe_load(f)
    test_img_dir = Path(dcfg["path"]) / dcfg[args.split]
    test_images = sorted(test_img_dir.glob("*"))[:50]  # sample of 50 is enough for a stable mean
    if not test_images:
        print(f"  no images found at {test_img_dir}")
        per_img_ms = []
    else:
        per_img_ms = []
        for p_img in test_images:
            t0 = time.perf_counter()
            _ = model.predict(source=str(p_img), conf=args.conf, imgsz=args.imgsz, verbose=False)
            per_img_ms.append((time.perf_counter() - t0) * 1000.0)

    # Summarize
    box = metrics.box
    lines = [
        "═══════════════════════════════════════════════════════════════",
        f"  Detector evaluation summary",
        "═══════════════════════════════════════════════════════════════",
        f"  model       : {model_path.relative_to(REPO_ROOT) if model_path.is_relative_to(REPO_ROOT) else model_path}",
        f"  dataset     : {data_yaml.relative_to(REPO_ROOT) if data_yaml.is_relative_to(REPO_ROOT) else data_yaml}",
        f"  split       : {args.split}",
        f"  imgsz       : {args.imgsz}",
        f"  conf gate   : {args.conf}",
        "",
        f"  ── Detection quality ──",
        f"  mAP@0.50              : {box.map50:.4f}",
        f"  mAP@0.50:0.95         : {box.map:.4f}",
        f"  precision (P)         : {box.mp:.4f}",
        f"  recall (R)            : {box.mr:.4f}",
        "",
        f"  ── Per-image inference time (n={len(per_img_ms)} sample) ──",
    ]
    if per_img_ms:
        import statistics
        lines += [
            f"  mean                  : {statistics.mean(per_img_ms):>6.1f} ms",
            f"  median                : {statistics.median(per_img_ms):>6.1f} ms",
            f"  p90                   : {sorted(per_img_ms)[int(len(per_img_ms) * 0.9)]:>6.1f} ms",
        ]
    else:
        lines.append("  (no test images found)")
    lines.append("═══════════════════════════════════════════════════════════════")

    summary = "\n".join(lines)
    print()
    print(summary)

    # Save report
    report_path = Path(args.report) if args.report else model_path.with_suffix(".eval.txt")
    report_path.write_text(summary + "\n")
    print(f"\n[done] summary saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
