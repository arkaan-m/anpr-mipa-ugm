"""
OCR evaluation: Character-Level Accuracy (CLA) and Plate-Level Accuracy (PLA).
Compares EasyOCR, Tesseract, and fused dual-OCR.
"""

import csv
import json
from pathlib import Path
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import numpy as np

from pipeline.ocr_engine import run_easyocr, run_tesseract
from pipeline.text_validator import validate_plate
from pipeline.cropper import preprocess_plate
import cv2


def character_level_accuracy(predicted: str, ground_truth: str) -> float:
    """
    CLA = 1 - (edit_distance / max_length)
    Returns value in [0, 1].
    """
    pred = predicted.replace(" ", "").upper()
    gt = ground_truth.replace(" ", "").upper()

    if not gt:
        return 1.0 if not pred else 0.0

    max_len = max(len(pred), len(gt))
    dist = levenshtein_distance(pred, gt)
    return 1.0 - (dist / max_len)


def plate_level_accuracy(predicted: str, ground_truth: str) -> bool:
    """PLA: exact match after normalization."""
    return predicted.replace(" ", "").upper() == ground_truth.replace(" ", "").upper()


def evaluate_ocr(
    test_data: list[dict],
    enable_preprocessing: bool = True,
) -> dict:
    """
    Evaluate OCR on test data.

    test_data: list of {"image_path": str, "ground_truth": str}

    Returns metrics for EasyOCR, Tesseract, and fused.
    """
    results = {
        "easyocr": {"cla_scores": [], "pla_matches": [], "texts": []},
        "tesseract": {"cla_scores": [], "pla_matches": [], "texts": []},
        "fused": {"cla_scores": [], "pla_matches": [], "texts": []},
    }

    for item in test_data:
        image = cv2.imread(item["image_path"])
        gt = item["ground_truth"]

        if image is None:
            continue

        # Optionally preprocess
        if enable_preprocessing:
            processed = preprocess_plate(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Run OCR engines
        easy_result = run_easyocr(processed)
        tess_result = run_tesseract(processed)

        # Validate
        easy_text, easy_valid = validate_plate(easy_result.text)
        tess_text, tess_valid = validate_plate(tess_result.text)

        # Fuse
        if easy_valid and tess_valid:
            fused_text = easy_text if easy_result.confidence >= tess_result.confidence else tess_text
        elif easy_valid:
            fused_text = easy_text
        elif tess_valid:
            fused_text = tess_text
        else:
            fused_text = easy_text if easy_result.confidence >= tess_result.confidence else tess_text

        # Calculate metrics
        for engine, text in [("easyocr", easy_text), ("tesseract", tess_text), ("fused", fused_text)]:
            cla = character_level_accuracy(text, gt)
            pla = plate_level_accuracy(text, gt)
            results[engine]["cla_scores"].append(cla)
            results[engine]["pla_matches"].append(pla)
            results[engine]["texts"].append(text)

    # Aggregate
    summary = {}
    for engine in results:
        scores = results[engine]
        n = len(scores["cla_scores"])
        summary[engine] = {
            "count": n,
            "avg_cla": np.mean(scores["cla_scores"]) if n else 0.0,
            "pla": np.mean(scores["pla_matches"]) if n else 0.0,
            "cla_std": np.std(scores["cla_scores"]) if n else 0.0,
        }

    return summary


def plot_ocr_comparison(summary: dict, save_path: str | None = None):
    """Bar chart comparing EasyOCR, Tesseract, and fused."""
    engines = list(summary.keys())
    cla_values = [summary[e]["avg_cla"] for e in engines]
    pla_values = [summary[e]["pla"] for e in engines]

    x = np.arange(len(engines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, cla_values, width, label="CLA", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, pla_values, width, label="PLA", color="#55a868")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("OCR Engine")
    ax.set_ylabel("Accuracy")
    ax.set_title("OCR Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in engines])
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def load_test_data(csv_path: str) -> list[dict]:
    """
    Load test data from CSV with columns: image_path, ground_truth
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "image_path": row["image_path"],
                "ground_truth": row["ground_truth"],
            })
    return data


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_ocr.csv"
    test_data = load_test_data(csv_path)

    print("=== With Preprocessing ===")
    summary_with = evaluate_ocr(test_data, enable_preprocessing=True)
    for engine, metrics in summary_with.items():
        print(f"  {engine}: CLA={metrics['avg_cla']:.4f}, PLA={metrics['pla']:.4f}")

    print("\n=== Without Preprocessing ===")
    summary_without = evaluate_ocr(test_data, enable_preprocessing=False)
    for engine, metrics in summary_without.items():
        print(f"  {engine}: CLA={metrics['avg_cla']:.4f}, PLA={metrics['pla']:.4f}")

    plot_ocr_comparison(summary_with, "evaluation/ocr_comparison.png")
