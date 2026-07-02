"""
Ablation study: test pipeline under different preprocessing configurations.
4 conditions: no preprocessing, CLAHE only, CLAHE+deblur, full pipeline.
"""

import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pipeline import PipelineConfig
from pipeline.pipeline_runner import run_pipeline


ABLATION_CONDITIONS = {
    "no_preprocessing": PipelineConfig(
        enable_clahe=False,
        enable_wiener=False,
        enable_deskew=False,
        enable_binarization=False,
        enable_denoise=False,
        enable_sharpen=False,
    ),
    "clahe_only": PipelineConfig(
        enable_clahe=True,
        enable_wiener=False,
        enable_deskew=False,
        enable_binarization=False,
        enable_denoise=False,
        enable_sharpen=False,
    ),
    "clahe_deblur": PipelineConfig(
        enable_clahe=True,
        enable_wiener=True,
        enable_deskew=False,
        enable_binarization=False,
        enable_denoise=False,
        enable_sharpen=False,
    ),
    "full_pipeline": PipelineConfig(
        enable_clahe=True,
        enable_wiener=True,
        enable_deskew=True,
        enable_binarization=True,
        enable_denoise=True,
        enable_sharpen=True,
    ),
}


def run_ablation(test_images: list[str], model_variant: str = "yolov8s") -> dict:
    """
    Run ablation study across all conditions.

    test_images: list of image file paths
    Returns: {condition_name: {metrics...}}
    """
    results = {}

    for condition_name, config in ABLATION_CONDITIONS.items():
        config.model_variant = model_variant
        print(f"\n{'='*50}")
        print(f"Condition: {condition_name}")
        print(f"{'='*50}")

        condition_results = {
            "total_images": len(test_images),
            "total_detections": 0,
            "total_ocr_success": 0,
            "processing_times": [],
            "plates_detected": [],
        }

        for img_path in test_images:
            try:
                output = run_pipeline(img_path, config)

                for frame in output.frames:
                    condition_results["processing_times"].append(frame.processing_time_ms)
                    condition_results["total_detections"] += len(frame.plates)

                    for plate in frame.plates:
                        if plate.final_text:
                            condition_results["total_ocr_success"] += 1
                            condition_results["plates_detected"].append(plate.final_text)

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

        # Summary stats
        times = condition_results["processing_times"]
        condition_results["avg_time_ms"] = np.mean(times) if times else 0
        condition_results["detection_rate"] = (
            condition_results["total_detections"] / max(len(test_images), 1)
        )
        condition_results["ocr_success_rate"] = (
            condition_results["total_ocr_success"] / max(condition_results["total_detections"], 1)
        )

        results[condition_name] = condition_results
        print(f"  Detections: {condition_results['total_detections']}")
        print(f"  OCR success: {condition_results['total_ocr_success']}")
        print(f"  Avg time: {condition_results['avg_time_ms']:.0f}ms")

    return results


def plot_ablation(results: dict, save_path: str | None = None):
    """Plot ablation study results."""
    conditions = list(results.keys())
    det_rates = [results[c]["detection_rate"] for c in conditions]
    ocr_rates = [results[c]["ocr_success_rate"] for c in conditions]
    avg_times = [results[c]["avg_time_ms"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    x = np.arange(len(conditions))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, det_rates, width, label="Detection Rate", color="#4c72b0")
    bars2 = ax1.bar(x + width / 2, ocr_rates, width, label="OCR Success Rate", color="#55a868")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_ylabel("Rate")
    ax1.set_title("Ablation: Detection & OCR Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, 1.2)

    # Processing time
    bars3 = ax2.bar(conditions, avg_times, color="#dd8452")
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{bar.get_height():.0f}ms", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Ablation: Processing Time")
    ax2.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    import glob

    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/images/test"
    test_images = sorted(glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.png"))

    if not test_images:
        print(f"No test images found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(test_images)} test images")
    results = run_ablation(test_images[:50])  # Limit for speed
    plot_ablation(results, "evaluation/ablation_results.png")

    # Save raw results
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "plates_detected"}
                 for k, v in results.items()}
    with open("evaluation/ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
