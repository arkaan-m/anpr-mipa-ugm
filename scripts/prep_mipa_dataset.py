"""
Local data-prep helper for the MIPA YOLOv8 finetune workflow.

Takes a Roboflow YOLOv8 export (either a .zip or an already-extracted
folder), validates its structure, splits if needed, generates a
data.yaml ready for ultralytics, and prints stats so you can sanity-check
the dataset before paying for Colab compute time.

Usage:
    python scripts/prep_mipa_dataset.py data/raw/mipa_roboflow_v1.zip
    python scripts/prep_mipa_dataset.py data/raw/mipa_unzipped/

Both invocations produce:
    data/processed/mipa_v1/
      train/images/*.jpg
      train/labels/*.txt
      val/...
      test/...
      data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median

try:
    from PIL import Image
except ImportError:
    print("[!] PIL/Pillow required: pip install Pillow", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_NAME = "mipa_v1"
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.7, 0.2, 0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Roboflow YOLOv8 export — .zip or directory")
    parser.add_argument("--name", default=DEFAULT_OUTPUT_NAME, help="Output dataset name under data/processed/")
    parser.add_argument("--reshuffle", action="store_true", help="Ignore Roboflow's split and redo 70/20/10 ourselves")
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        print(f"[!] not found: {src}", file=sys.stderr)
        return 2

    out_dir = REPO_ROOT / "data" / "processed" / args.name
    if out_dir.exists():
        print(f"[!] output dir already exists, refusing to overwrite: {out_dir}")
        print(f"    rm -rf {out_dir.relative_to(REPO_ROOT)}  # to redo")
        return 2

    work_dir = _unpack_if_needed(src)
    print(f"[load] source: {work_dir}")

    # Roboflow YOLOv8 exports have either:
    #  - train/images, train/labels, valid/images, valid/labels, test/images, test/labels, data.yaml
    #  - or a single images/ + labels/ folder if unsplit
    pairs_by_split = _discover_pairs(work_dir, args.reshuffle)

    total = sum(len(v) for v in pairs_by_split.values())
    if total == 0:
        print("[!] no (image, label) pairs found. Did the export include labels?", file=sys.stderr)
        return 1

    print(f"[copy] writing to {out_dir.relative_to(REPO_ROOT)}/")
    for split in SPLITS:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs_by_split[split]:
            shutil.copy2(img_path, out_dir / split / "images" / img_path.name)
            shutil.copy2(lbl_path, out_dir / split / "labels" / lbl_path.name)

    _write_data_yaml(out_dir)
    _print_stats(out_dir)
    print(f"[done] {out_dir / 'data.yaml'}")
    print("       upload this folder to Google Drive at:")
    print("       /MyDrive/ANPR_MIPA_UGM/datasets/" + args.name + "/")
    return 0


# ─────────────────────────── helpers ──────────────────────────

def _unpack_if_needed(src: Path) -> Path:
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        raise SystemExit(f"unsupported source type: {src}")

    extract_to = REPO_ROOT / "data" / "processed" / "_extracted" / src.stem
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] extracting to {extract_to.relative_to(REPO_ROOT)}/")
    with zipfile.ZipFile(src) as z:
        z.extractall(extract_to)
    return extract_to


def _discover_pairs(root: Path, reshuffle: bool) -> dict[str, list[tuple[Path, Path]]]:
    """Find all (image, label) pairs grouped by split.

    Handles two Roboflow layouts:
      A) train/images,  train/labels, valid/images, valid/labels, test/images, test/labels
      B) images/, labels/  (unsplit)
    """
    # Normalize 'valid' → 'val'
    by_split: dict[str, list[tuple[Path, Path]]] = {s: [] for s in SPLITS}

    split_dirs = {
        "train": ["train"],
        "val": ["val", "valid"],
        "test": ["test"],
    }

    has_any_split = any(
        any((root / d / "images").is_dir() for d in dirs)
        for dirs in split_dirs.values()
    )

    if has_any_split and not reshuffle:
        for canonical, candidates in split_dirs.items():
            for cand in candidates:
                img_dir = root / cand / "images"
                lbl_dir = root / cand / "labels"
                if img_dir.is_dir() and lbl_dir.is_dir():
                    by_split[canonical].extend(_pair_dirs(img_dir, lbl_dir))
        return by_split

    # Unsplit layout — or user asked to reshuffle
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not (img_dir.is_dir() and lbl_dir.is_dir()):
        # Last resort: scan recursively for image/label twins
        all_pairs: list[tuple[Path, Path]] = []
        for img in root.rglob("*.[jJ][pP][gG]"):
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                all_pairs.append((img, lbl))
    else:
        all_pairs = _pair_dirs(img_dir, lbl_dir)

    return _split_pairs(all_pairs)


def _pair_dirs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"[warn] no label for {img.name}")
    return pairs


def _split_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    random.seed(42)
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


def _write_data_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['license_plate']\n"
    )


def _print_stats(out_dir: Path) -> None:
    print("[stats]")
    for split in SPLITS:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        imgs = list(img_dir.glob("*"))
        n_box = 0
        sizes: list[tuple[int, int]] = []
        aspects: list[float] = []
        area_pcts: list[float] = []
        bad_class: list[str] = []
        for img in imgs:
            try:
                with Image.open(img) as p:
                    w, h = p.size
                    sizes.append((w, h))
            except Exception:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = parts[0]
                if cls != "0":
                    bad_class.append(f"{img.name}: class={cls}")
                _, _, _, bw, bh = parts
                bw_f, bh_f = float(bw), float(bh)
                n_box += 1
                if bh_f > 0:
                    aspects.append(bw_f / bh_f)
                area_pcts.append(bw_f * bh_f * 100)
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        wmed = median(widths) if widths else 0
        hmed = median(heights) if heights else 0
        amed = median(aspects) if aspects else 0
        pmed = median(area_pcts) if area_pcts else 0
        print(f"        {split:>5}: {len(imgs):>4} images, {n_box:>4} boxes,  "
              f"median size={wmed}×{hmed},  aspect={amed:.2f},  area={pmed:.2f}%")
        if bad_class:
            print(f"               [warn] {len(bad_class)} labels with non-zero class id (first few: {bad_class[:3]})")


if __name__ == "__main__":
    sys.exit(main())
