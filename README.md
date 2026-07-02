# ANPR System — MIPA UGM Parking Lot

Automatic License Plate Recognition system for the MIPA Faculty parking lot at Universitas Gadjah Mada, built as an undergraduate thesis (Skripsi Produk).

Users photograph a vehicle, upload via a Flask web interface, and the system returns **AUTHORIZED / UNAUTHORIZED / UNCERTAIN**.

**Author:** Arkaan Muhammad — Ilmu Komputer, Universitas Gadjah Mada  
**Supervisor:** Prof. Dra. Sri Hartati, M.Sc., Ph.D.

---

## Links

| Resource | URL |
|----------|-----|
| **Dataset (MIPA + public)** | [Google Drive — ANPR_MIPA_UGM/datasets/](*FILL_IN_DRIVE_LINK*) |
| **Trained model weights** | [Google Drive — ANPR_MIPA_UGM/models/](*FILL_IN_DRIVE_LINK*) |
| **Thesis draft (proposal + results)** | [Google Drive — ANPR_MIPA_UGM/docs/](*FILL_IN_DRIVE_LINK*) |
| **This repository** | [github.com/YOUR_USERNAME/anpr-mipa-ugm](*FILL_IN_GITHUB_LINK*) |

---

## System overview

```
Photo upload → EXIF fix → YOLOv8s detection → plate crop →
EasyOCR + Tesseract (parallel) → text normalisation →
SQLite lookup (exact + fuzzy Levenshtein ≤2) → verdict
```

**Stack:** Python 3.11 · Flask · YOLOv8 (Ultralytics) · EasyOCR · Tesseract · SQLite

---

## Key results (evaluated on 30-image curated MIPA test set)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| System accuracy | 93.3% | ≥ 75% | ✓ |
| False Acceptance Rate | 0.0% | < 10% | ✓ |
| False Rejection Rate | 0.0% | < 15% | ✓ |
| Verification F1 | 88.9% | ≥ 75% | ✓ |
| OCR character-level accuracy (CLA) | 85.4% | ≥ 85% | ✓ |
| Processing time (warm) | ~3.2 s/img | < 5 s | ✓ |
| Detection mAP@0.5 (test) | 0.839 | ≥ 80% | ✓ |
| Detection mAP@0.5 (val) | 0.823 | ≥ 85% | ✗ |
| mAP@0.5:0.95 | ~0.54 | ≥ 60% | ✗ |
| OCR plate-level accuracy (PLA) | 50.0% | ≥ 80% | ✗ |

Detection model: **YOLOv8s**, two-stage transfer learning (COCO → public Indonesian dataset → MIPA).

---

## Repository structure

```
├── app.py                  # Flask entry point (port 5001)
├── config.py               # config.yaml loader
├── config.yaml             # all tunables (thresholds, model, OCR flags)
├── init_db.py              # create SQLite schema + seed authorized plates
├── requirements.txt
├── pipeline/
│   ├── detector.py         # YOLOv8 inference
│   ├── ocr_engine.py       # dual EasyOCR + Tesseract, parallel
│   ├── plate_extractor.py  # crop, deskew, binarize
│   ├── preprocessor.py     # letterbox 640×640, CLAHE (disabled by default)
│   ├── text_normalizer.py  # regex, Indonesian region-code validation
│   ├── verifier.py         # SQLite lookup + confidence gating
│   ├── logger.py           # CSV + DB logging
│   └── pipeline.py         # orchestrator
├── scripts/
│   ├── eval_detector.py    # mAP50 / recall (conf=0.001, standard)
│   ├── eval_ocr.py         # CLA / PLA per engine
│   ├── eval_system.py      # end-to-end accuracy + FAR/FRR + CI
│   ├── eval_ablation.py    # preprocessing ablation (4 conditions)
│   ├── eval_variants.py    # YOLOv8 n/s/m variant comparison
│   ├── auto_label_mipa.py  # auto-label raw photos → YOLO .txt
│   ├── review_labels.py    # browser label reviewer (localhost:7777)
│   └── prep_mipa_local.py  # 70/15/15 split → data/processed/mipa_v1/
├── notebooks/
│   ├── 05_mipa_finetune.ipynb      # Colab: YOLOv8s fine-tune (single-stage, historical)
│   ├── 06_colab_app.ipynb          # Colab: run full Flask app via ngrok
│   └── 07_variant_comparison.ipynb # Colab: two-stage n/s/m training + comparison
├── data/
│   ├── anonymized/             # ← plate-masked CSVs (published in this repo)
│   │   ├── ocr_gt.csv          #   OCR ground truth (plates masked: AB #### XT)
│   │   ├── system_gt.csv       #   system ground truth (plates masked)
│   │   ├── ocr_eval_results.csv
│   │   └── system_eval_results.csv
│   ├── ablation_results.csv    # preprocessing ablation (metrics only, no PII)
│   └── variant_comparison_final.csv  # variant comparison (metrics only, no PII)
├── models/                     # ← weights NOT included (see Drive link above)
│   ├── best_yolov8s.pt         # DEPLOYED — tuned two-stage
│   ├── best_yolov8n.pt         # comparison variant
│   └── best_yolov8m.pt         # comparison variant
├── docs/
│   └── TRAINING.md             # 6-phase training guide
└── static/js/upload.js         # dropzone + progress UI
```

> **Model weights** (`*.pt`) are excluded from this repository due to file size. Download from the Google Drive link above and place in `models/`.

---

## Setup

**Requirements:** Python 3.11, Tesseract OCR installed system-wide.

```bash
# Install Tesseract (macOS)
brew install tesseract

# Clone and install Python dependencies
git clone https://github.com/YOUR_USERNAME/anpr-mipa-ugm.git
cd anpr-mipa-ugm
python -m venv venv
venv/bin/pip install -r requirements.txt

# Download model weights from Drive → place in models/
# e.g. models/best_yolov8s.pt

# Initialise the database (creates anpr_mipa.db + seeds AB 1194 XT)
venv/bin/python init_db.py

# Run the app
venv/bin/python app.py
# → open http://localhost:5001
```

---

## Running evaluations

> **Important:** the 30-image test set lives in `data/processed/mipa_v1/test/images/`.  
> Always pass `--images-dir` — the default raw/ directory does not contain the full test set.

```bash
# Detection mAP (standard conf=0.001)
venv/bin/python scripts/eval_detector.py --model models/best_yolov8s.pt \
    --data data/processed/mipa_v1/data.yaml --split val

# OCR accuracy (CLA / PLA)
venv/bin/python scripts/eval_ocr.py \
    --gt data/ocr_gt.csv \
    --images-dir data/processed/mipa_v1/test/images

# End-to-end system accuracy + FAR/FRR
venv/bin/python scripts/eval_system.py \
    --gt data/system_gt.csv \
    --images-dir data/processed/mipa_v1/test/images

# Preprocessing ablation
venv/bin/python scripts/eval_ablation.py \
    --gt data/system_gt.csv \
    --images-dir data/processed/mipa_v1/test/images

# YOLOv8 variant comparison (n / s / m)
venv/bin/python scripts/eval_variants.py
```

---

## Data privacy note

The plate images and full unmasked ground-truth data are **not** committed to this public repository, to protect the privacy of vehicle owners photographed in the parking lot. Specifically:

- **Plate images** (`data/raw/`, `data/processed/`) — hosted on the Google Drive link above, not in git.
- **Ground-truth & result CSVs** — the copies in `data/anonymized/` have every plate's digits masked (`AB 1194 XT` → `AB #### XT`). The unmasked originals are on Drive.
- **Database** (`anpr_mipa.db`) — not committed (regenerate with `init_db.py`); the seed owner name has been genericized.

The masked CSVs preserve the structure and per-row metrics (CLA, PLA, confidence, verdict) for inspection; exact-match reproduction requires the unmasked data from Drive.

## Dataset

The MIPA dataset was collected at the MIPA UGM parking lot and is **not** included in this repository (privacy of vehicle owners).

| Split | Images | Labeled plates |
|-------|--------|----------------|
| Train | 204 | 204 |
| Val | 43 | 59 |
| Test (full) | 45 | 45 |
| Test (curated) | 30 | 30 |

Download the labeled dataset from the Google Drive link at the top of this file.

The public pre-training dataset is the [Indonesian License Plate dataset on Kaggle](https://www.kaggle.com/datasets/imamdigmi/indonesian-plate-number) (958/274/138 split, used for Stage-1 transfer learning).

---

## Authorized plates (database seed)

| Plate | Owner |
|-------|-------|
| AB 1194 XT | Faculty vehicle (test subject) |

Add more via `python init_db.py` (after editing the seeds list) or direct SQL.

---

## License

This project was developed for academic purposes at Universitas Gadjah Mada.
