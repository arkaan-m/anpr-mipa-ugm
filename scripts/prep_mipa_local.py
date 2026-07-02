"""
Package the locally-annotated MIPA photos into a YOLO training dataset.

Reads images + .txt label files from data/raw/mipa_photos/, splits
70/15/15 train/val/test, and writes to data/processed/mipa_v1/.

Usage:
    venv/bin/python scripts/prep_mipa_local.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIR = REPO_ROOT / "data" / "raw" / "mipa_photos"
OUT_DIR   = REPO_ROOT / "data" / "processed" / "mipa_v1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
SPLIT = (0.70, 0.15, 0.15)


def main() -> int:
    if OUT_DIR.exists():
        print(f"[!] output already exists: {OUT_DIR.relative_to(REPO_ROOT)}")
        print(f"    rm -rf {OUT_DIR.relative_to(REPO_ROOT)}  # to redo")
        return 2

    # Collect only images that have a non-empty label file
    pairs: list[tuple[Path, Path]] = []
    skipped = []
    for img in sorted(PHOTO_DIR.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = img.with_suffix(".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            skipped.append(img.name)
            continue
        # Skip classes.txt accidentally named as image pair
        content = lbl.read_text().strip()
        if not content or content.startswith("license_plate"):
            skipped.append(img.name)
            continue
        pairs.append((img, lbl))

    if not pairs:
        print("[!] no labeled images found", file=sys.stderr)
        return 2

    print(f"[load] {len(pairs)} labeled images  ({len(skipped)} skipped — no label)")
    if skipped:
        for s in skipped:
            print(f"       skipped: {s}")

    # Split
    random.seed(SEED)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    # Write
    for split, items in splits.items():
        img_dir = OUT_DIR / split / "images"
        lbl_dir = OUT_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)
        print(f"  {split:>5}: {len(items)} images")

    # data.yaml
    yaml_path = OUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "test:  test/images\n"
        "nc: 1\n"
        "names: ['license_plate']\n"
    )

    print(f"\n[done] dataset at {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"       data.yaml: {yaml_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
