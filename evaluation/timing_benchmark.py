"""
Timing benchmark: per-stage latency analysis.
Verifies the <5 second per-image target.
"""

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import MODELS_DIR, MODEL_VARIANTS, TARGET_PROCESSING_TIME_MS
from pipeline import PipelineConfig
from pipeline.preprocessor import preprocess
from pipeline.detector import PlateDetector
from pipeline.cropper import crop_plate, preprocess_plate
from pipeline.ocr_engine import run_easyocr, run_tesseract
from pipeline.text_validator import validate_plate


def benchmark_single_image(image_path: str, model_variant: str = "yolov8s") -> dict:
    """
    Benchmark each pipeline stage on a single image.
    Returns timing dict in milliseconds.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    timings = {}

    # Stage 1: Preprocessing
    t0 = time.time()
    preprocessed, scale_info = preprocess(image)
    timings["preprocessing"] = (time.time() - t0) * 1000

    # Stage 2: Detection
    detector = PlateDetector(model_variant)
    t0 = time.time()
    detections = detector.detect(preprocessed, scale_info)
    timings["detection"] = (time.time() - t0) * 1000

    timings["num_detections"] = len(detections)

    if detections:
        det = detections[0]

        # Stage 3: Cropping + plate preprocessing
        t0 = time.time()
        crop = crop_plate(image, det)
        processed_crop = preprocess_plate(crop)
        timings["crop_preprocess"] = (time.time() - t0) * 1000

        # Stage 4a: EasyOCR
        t0 = time.time()
        easy_result = run_easyocr(processed_crop)
        timings["easyocr"] = (time.time() - t0) * 1000

        # Stage 4b: Tesseract
        t0 = time.time()
        tess_result = run_tesseract(processed_crop)
        timings["tesseract"] = (time.time() - t0) * 1000

        # Stage 5: Validation
        t0 = time.time()
        validate_plate(easy_result.text)
        validate_plate(tess_result.text)
        timings["validation"] = (time.time() - t0) * 1000
    else:
        timings["crop_preprocess"] = 0
        timings["easyocr"] = 0
        timings["tesseract"] = 0
        timings["validation"] = 0

    timings["total"] = sum(v for k, v in timings.items() if k != "num_detections")
    timings["meets_target"] = timings["total"] < TARGET_PROCESSING_TIME_MS

    return timings


def benchmark_batch(image_paths: list[str], model_variant: str = "yolov8s") -> dict:
    """Benchmark across multiple images."""
    all_timings = []

    for path in image_paths:
        try:
            t = benchmark_single_image(path, model_variant)
            all_timings.append(t)
        except Exception as e:
            print(f"Error benchmarking {path}: {e}")

    if not all_timings:
        return {}

    # Aggregate
    stages = ["preprocessing", "detection", "crop_preprocess", "easyocr", "tesseract", "validation", "total"]
    summary = {}
    for stage in stages:
        values = [t[stage] for t in all_timings]
        summary[stage] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
        }

    meets_target = sum(1 for t in all_timings if t["meets_target"])
    summary["target_compliance"] = meets_target / len(all_timings)

    return summary


def plot_timing(summary: dict, save_path: str | None = None):
    """Plot per-stage timing breakdown."""
    stages = ["preprocessing", "detection", "crop_preprocess", "easyocr", "tesseract", "validation"]
    means = [summary[s]["mean"] for s in stages]
    stds = [summary[s]["std"] for s in stages]
    labels = ["Preprocess", "Detection", "Crop+OCR\nPreprocess", "EasyOCR", "Tesseract", "Validation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    bars = ax1.bar(labels, means, yerr=stds, color=colors, capsize=4)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{mean:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Per-Stage Timing Breakdown")
    ax1.axhline(y=TARGET_PROCESSING_TIME_MS, color="red", linestyle="--", alpha=0.5, label=f"Target: {TARGET_PROCESSING_TIME_MS}ms")

    # Pie chart
    ax2.pie(means, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"Processing Time Distribution\n(Total: {summary['total']['mean']:.0f}ms avg)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Print summary
    print(f"\nTotal: {summary['total']['mean']:.0f}ms avg "
          f"(min={summary['total']['min']:.0f}, max={summary['total']['max']:.0f})")
    print(f"Target compliance (<{TARGET_PROCESSING_TIME_MS}ms): "
          f"{summary['target_compliance']*100:.1f}%")


if __name__ == "__main__":
    import sys
    import glob

    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/images/test"
    images = sorted(glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.png"))[:20]

    if not images:
        print(f"No images found in {test_dir}")
        sys.exit(1)

    print(f"Benchmarking {len(images)} images...")
    summary = benchmark_batch(images)
    plot_timing(summary, "evaluation/timing_benchmark.png")
