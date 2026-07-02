"""
Auto-label MIPA parking lot photos using the current YOLOv8 model.

Runs inference on every image in an input folder and saves YOLO-format
.txt label files alongside each image. Images where the model finds no
plate are listed at the end so you know which ones need manual boxes.

Usage:
    venv/bin/python scripts/auto_label_mipa.py
    venv/bin/python scripts/auto_label_mipa.py --input data/raw/mipa_photos --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/raw/mipa_photos", help="folder of raw photos")
    p.add_argument("--model", default="models/best_yolov8s.pt", help="YOLOv8 weights")
    p.add_argument("--conf", type=float, default=0.20, help="detection confidence (lower = more recalls, more noise)")
    p.add_argument("--imgsz", type=int, default=1280, help="inference image size")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip images that already have a .txt label (preserves manual fixes)")
    args = p.parse_args()

    input_dir = REPO_ROOT / args.input
    model_path = REPO_ROOT / args.model

    if not input_dir.is_dir():
        print(f"[!] input folder not found: {input_dir}", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"[!] model not found: {model_path}", file=sys.stderr)
        return 2

    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[!] no images found in {input_dir}", file=sys.stderr)
        return 2

    print(f"[load] model  = {model_path.relative_to(REPO_ROOT)}")
    print(f"[load] input  = {input_dir.relative_to(REPO_ROOT)}")
    print(f"[load] images = {len(images)}")
    print(f"[load] conf   = {args.conf}  imgsz = {args.imgsz}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    labeled = []
    skipped = []

    preserved = []
    for img_path in images:
        if args.skip_existing and img_path.with_suffix(".txt").exists():
            preserved.append(img_path.name)
            continue
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            skipped.append(img_path.name)
            print(f"  [no det]  {img_path.name}")
            continue

        # Write YOLO label file: class_id x_center y_center width height (normalized)
        label_path = img_path.with_suffix(".txt")
        ih, iw = result.orig_shape
        lines = []
        for box in boxes.xyxy.tolist():
            x1, y1, x2, y2 = box
            xc = ((x1 + x2) / 2) / iw
            yc = ((y1 + y2) / 2) / ih
            bw = (x2 - x1) / iw
            bh = (y2 - y1) / ih
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines))
        conf_scores = [f"{c:.2f}" for c in boxes.conf.tolist()]
        labeled.append(img_path.name)
        print(f"  [labeled] {img_path.name}  boxes={len(lines)}  conf={conf_scores}")

    print()
    print("═" * 60)
    print(f"  Auto-labeled : {len(labeled)} / {len(images)} images")
    print(f"  Preserved    : {len(preserved)} images (already had labels, skipped)")
    print(f"  No detection : {len(skipped)} images (need manual boxes)")
    if skipped:
        print()
        print("  Files needing manual annotation:")
        for name in skipped:
            print(f"    {name}")
    print("═" * 60)
    print()
    print("Next step: install labelImg and review/fix the boxes.")
    print("  pip install labelImg")
    print(f"  labelImg {input_dir} {input_dir}/classes.txt")
    print()
    print("  In labelImg: View → Auto Save, format = YOLO")
    print("  Keyboard shortcuts: A/D = prev/next, W = draw box, Del = delete box")

    # Write classes.txt for labelImg
    classes_path = input_dir / "classes.txt"
    classes_path.write_text("license_plate\n")
    print(f"\n[done] classes.txt written to {input_dir.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
