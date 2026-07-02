# Training the YOLOv8 detector on MIPA-specific data

This guide takes you from "I have a phone" to "I have a model that detects
plates at MIPA at 0.85+ confidence on the spec's default thresholds."

## Why this matters

The current model (`models/best_yolov8s.pt`) was trained on generic Indonesian
plate datasets — close-up, well-lit, mostly straight-on shots. On your actual
MIPA deployment conditions (dim parking garage, wide-angle phone photos,
oblique angles, plates 50–100 px wide in a 5712×4284 frame), it detects at
**0.35–0.50 confidence**. We currently compensate by:

- Letterboxing at 1280×1280 (4× the spec's 640) — costs ~2× inference time
- Lowering the detection-confidence floor to 0.30 (vs spec's 0.50) — accepts more candidates

Both work, but they're band-aids. A finetune on ~200 MIPA-specific images
should let you restore the spec defaults (640 imgsz, 0.5 conf) and *still*
detect at ~0.9 confidence on your real photos.

---

## Phase 1 — Collect photos (2–4 hours)

**Target: 200–300 photos.** Quality over quantity, but variety matters.

Cover all of these (rough distribution):

| Condition | Count | Notes |
|---|---|---|
| Morning sun | ~50 | natural light, sharp shadows |
| Midday overcast | ~50 | diffuse light, low contrast |
| Late afternoon | ~50 | warm/golden light, sometimes glare |
| Evening / dim | ~50 | parking garage lighting only |
| Rainy / wet plates | ~30 | reflections, droplets |
| Far / wide angle | ~30 | plate <100 px wide — like IMG_3246 |
| Negative frames | ~20 | empty parking spaces, just buildings — teaches the model what is NOT a plate |

For each condition, mix:
- **Vehicle types**: ~70% cars, ~30% motorcycles
- **Plate orientation**: 70% straight-on, 30% angled ±15°
- **Distance**: 50% close (plate >300 px wide), 50% far (plate 50–200 px)

Drop all photos into a single folder, e.g. `data/raw/mipa_2026/`. Don't worry
about renaming or organizing — Roboflow handles that.

---

## Phase 2 — Label with Roboflow (3–5 hours)

Roboflow Universe has a free tier that's enough for this.

1. Go to https://app.roboflow.com → create a new project
   - Project type: **Object Detection**
   - Annotation group: `license-plate`
   - License: pick whichever
2. Upload all 200–300 photos
3. **Label by drawing tight bounding boxes around the plate text region only.**
   This is critical — *do not* include the plate's metal carrier, year stickers,
   dealer ads, or vehicle frame. We learned the hard way that loose boxes
   cause OCR to read the dealer ad. The tighter, the better. One box per plate.
4. Use Roboflow's "Auto-label" only if you verify every result — it's wrong on
   ~10% of MIPA-style photos.
5. **Apply augmentations** in Roboflow's preprocessing step:
   - Auto-orient (handles EXIF — same fix we did in `input_handler.py`)
   - Resize to 1280×1280 (matches our inference setting)
6. **Apply augmentations** in the augmentation step:
   - Brightness ±20%
   - Saturation ±15%
   - Blur up to 1.5 px
   - Rotation ±5° (don't go higher; our pipeline already deskews)
   - Generate 3× per image → ~600–900 augmented training images
7. **Generate dataset version**, format: **YOLOv8**
8. Download the ZIP. Save it as `data/raw/mipa_roboflow_v1.zip`.

---

## Phase 3 — Local prep (5 minutes)

Validate the export before paying for Colab compute time.

```bash
venv/bin/python scripts/prep_mipa_dataset.py data/raw/mipa_roboflow_v1.zip
```

You'll see:

```
[unzip] extracted to data/processed/mipa_v1/
[stats]   train: 600 images, 612 boxes
            val:  90 images, 92 boxes
           test:  60 images, 61 boxes
[stats] image sizes (min/median/max): 1280×1280
[stats] box aspect ratio (median): 2.95
[stats] box area % of image (median): 1.2%
[done] data.yaml written to data/processed/mipa_v1/data.yaml
```

If you see warnings ("3 images have no labels", "label class ID > 0"), fix
them in Roboflow and re-export before training.

---

## Phase 4 — Finetune on Colab (30–60 minutes runtime)

1. Open `notebooks/05_mipa_finetune.ipynb` in Colab (Runtime → Change runtime
   type → **T4 GPU**)
2. Upload `data/processed/mipa_v1/` as a zip to your Google Drive at
   `/MyDrive/ANPR_MIPA_UGM/datasets/mipa_v1/`
3. Run all cells. The notebook:
   - Starts from `yolov8s.pt` (general-purpose plate detector)
   - Finetunes with a low learning rate (transfer learning, not from-scratch)
   - Trains for 50 epochs at `imgsz=1280` (matches our inference)
   - Saves `best.pt` to `/MyDrive/ANPR_MIPA_UGM/weights/best_yolov8s_mipa.pt`
4. Download `best_yolov8s_mipa.pt` from Drive

Expected training metrics:
- mAP50 > 0.90 on MIPA validation set
- mAP50-95 > 0.65

If you're below those, your dataset is too small or has labeling
inconsistencies. Don't deploy.

---

## Phase 5 — Deploy & restore spec defaults

1. Move the new weights into place:
   ```bash
   mv ~/Downloads/best_yolov8s_mipa.pt models/best_yolov8s.pt
   ```
2. In `config.yaml`, restore spec defaults:
   ```yaml
   detection:
     confidence_threshold: 0.5    # was 0.30 — the new model is confident
   preprocessing:
     input_size: [640, 640]       # was [1280, 1280] — back under 5s
   ```
3. Restart Flask (or just save `config.yaml` — Werkzeug will reload it).
4. Verify with `curl http://127.0.0.1:5001/healthz` — should show `0.5` and `[640, 640]`.

---

## Phase 6 — Validate it actually improved (15 minutes)

Re-upload the three test images:
- `IMG_4164.JPG` (AB 1194 XT — closeup, was the easy case)
- `IMG_4165.JPG` (AA 1779 NF — far, was at 0.91 with 1280)
- `IMG_3246.JPG` (B 1289 FAC — dim/angled, was at 0.47 with 1280)

Expected results after MIPA finetune:
- All three detect at **>0.85 confidence**
- All three pass the spec's 0.5 threshold easily
- Processing time per image: **~2–3 seconds** (down from 6–12s at 1280)

If any drop, it means your training data didn't cover that condition. Add
more photos in that category and re-train.

---

## Same playbook for OCR

If you want to go further: crop the labeled plates (use `plate_extractor.py`
to batch-export crops with their text labels) and finetune EasyOCR's
recognition head on those crops. That fixes the `4→L` family of confusions at
the OCR layer instead of the substitution layer, and pushes OCR confidence
from ~0.55 to ~0.85 — which closes the last gap to AUTHORIZED on
borderline reads.

This is the topic of `notebooks/04_ocr_benchmark.ipynb`. Same pattern: collect,
label (text strings this time), train, replace.
