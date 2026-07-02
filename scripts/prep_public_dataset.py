"""
Extract + convert the public Indonesian plate detection dataset
(`linkgish/indonesian-plate-number-from-multi-sources` on Kaggle) into
YOLOv8 training format.

Input:
    data/indonesian-plate-number-from-multi-sources.zip

Pipeline:
    1. Extract only plate_detection_dataset/ (the OCR sub-dataset is
       a separate concern — we'll handle it in scripts/prep_ocr_dataset.py)
    2. Parse annotations.json (COCO format) and convert bboxes to YOLO
       format (class_id x_center y_center width height, all normalized 0–1)
    3. Split into train/val/test (70/20/10) with a fixed seed
    4. Write data.yaml ready for ultralytics

Output:
    data/processed/public_v1/
        train/images/*.png|jpg
        train/labels/*.txt
        val/...
        test/...
        data.yaml

This is the BASELINE dataset for Phase 1 of the training plan. The MIPA-
specific finetune in Phase 3 starts from a model trained on this data.

Usage:
    venv/bin/python scripts/prep_public_dataset.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data" / "indonesian-plate-number-from-multi-sources.zip"
OUT_NAME = "public_v1"
OUT_DIR = REPO_ROOT / "data" / "processed" / OUT_NAME
EXTRACT_DIR = REPO_ROOT / "data" / "_extracted" / "public_indonesian"
SPLIT_RATIOS = (0.70, 0.20, 0.10)
SPLITS = ("train", "val", "test")
SEED = 42

DETECTION_ROOT_IN_ZIP = "plate_detection_dataset/plate_detection_dataset"


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"[!] dataset zip not found at {ZIP_PATH}", file=sys.stderr)
        return 2

    if OUT_DIR.exists():
        print(f"[!] output already exists: {OUT_DIR.relative_to(REPO_ROOT)}")
        print(f"    rm -rf {OUT_DIR.relative_to(REPO_ROOT)}  # to redo")
        return 2

    _extract_detection_only()
    coco_path, image_dir = _find_coco_artifacts()
    print(f"[parse] COCO: {coco_path.relative_to(REPO_ROOT)}")
    print(f"[parse] images: {image_dir.relative_to(REPO_ROOT)}")

    pairs = _convert_coco_to_yolo(coco_path, image_dir)
    print(f"[convert] generated {len(pairs)} (image, label) pairs")

    splits = _split(pairs)
    _write_splits(splits)
    _write_data_yaml()
    _print_stats(splits)
    print(f"[done] data.yaml at {(OUT_DIR / 'data.yaml').relative_to(REPO_ROOT)}")
    return 0


# ──────────────────────────── extract ──────────────────────────

def _extract_detection_only() -> None:
    """Extract only the detection sub-dataset from the bundled zip."""
    if EXTRACT_DIR.exists():
        print(f"[extract] already extracted at {EXTRACT_DIR.relative_to(REPO_ROOT)}")
        return
    print(f"[extract] extracting plate_detection_dataset/ to {EXTRACT_DIR.relative_to(REPO_ROOT)}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = [n for n in z.namelist() if n.startswith("plate_detection_dataset/")]
        for i, n in enumerate(names):
            if i % 100 == 0:
                print(f"  ... {i}/{len(names)}", end="\r", flush=True)
            z.extract(n, EXTRACT_DIR)
        print(f"  extracted {len(names)} entries     ")


# ───────────────────────── COCO → YOLO ─────────────────────────

def _find_coco_artifacts() -> tuple[Path, Path]:
    coco = EXTRACT_DIR / DETECTION_ROOT_IN_ZIP / "annotations" / "annotations.json"
    images = EXTRACT_DIR / DETECTION_ROOT_IN_ZIP / "images"
    if not coco.exists():
        raise SystemExit(f"COCO JSON missing: {coco}")
    if not images.is_dir():
        raise SystemExit(f"images dir missing: {images}")
    return coco, images


def _convert_coco_to_yolo(coco_path: Path, image_dir: Path) -> list[tuple[Path, str]]:
    """Returns a list of (image_path, yolo_label_text) pairs.

    The dataset is single-class (`plate_number` → class_id 0).
    COCO bbox: [x, y, w, h] in absolute pixels (top-left + dims).
    YOLO bbox: [class_id, x_center, y_center, w, h] all normalized to [0, 1].
    """
    with coco_path.open() as f:
        coco = json.load(f)

    image_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    boxes_by_image: dict[int, list[list[float]]] = {}
    for ann in coco["annotations"]:
        boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    pairs: list[tuple[Path, str]] = []
    skipped_missing = 0
    skipped_no_box = 0
    for img_id, img in image_by_id.items():
        fname = img["file_name"]
        img_path = image_dir / fname
        if not img_path.exists():
            skipped_missing += 1
            continue
        boxes = boxes_by_image.get(img_id, [])
        if not boxes:
            skipped_no_box += 1
            continue

        # Width/height stored as strings in this dataset — coerce
        w_img = float(img["width"])
        h_img = float(img["height"])
        if w_img <= 0 or h_img <= 0:
            skipped_no_box += 1
            continue

        lines: list[str] = []
        for bx, by, bw, bh in boxes:
            xc = (bx + bw / 2) / w_img
            yc = (by + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            # Clamp into [0, 1] in case the source is slightly out of bounds
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        pairs.append((img_path, "\n".join(lines)))

    if skipped_missing:
        print(f"  [warn] {skipped_missing} COCO entries reference missing image files")
    if skipped_no_box:
        print(f"  [warn] {skipped_no_box} images had no usable annotations")
    return pairs


# ───────────────────────────── split ───────────────────────────

def _split(pairs: list[tuple[Path, str]]) -> dict[str, list[tuple[Path, str]]]:
    random.seed(SEED)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _write_splits(splits: dict[str, list[tuple[Path, str]]]) -> None:
    print("[write]")
    for split, pairs in splits.items():
        img_dir = OUT_DIR / split / "images"
        lbl_dir = OUT_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_src, label_text in pairs:
            dest_img = img_dir / img_src.name
            dest_lbl = lbl_dir / (img_src.stem + ".txt")
            shutil.copy2(img_src, dest_img)
            dest_lbl.write_text(label_text)
        print(f"  {split}: {len(pairs)} images")


def _write_data_yaml() -> None:
    yaml_path = OUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['license_plate']\n"
    )


def _print_stats(splits: dict[str, list[tuple[Path, str]]]) -> None:
    print("[stats]")
    for split, pairs in splits.items():
        n_box = sum(len(lt.splitlines()) for _, lt in pairs)
        print(f"  {split:>5}: {len(pairs):>4} images, {n_box:>4} bboxes")


if __name__ == "__main__":
    sys.exit(main())
