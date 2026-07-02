"""
Detection evaluation: mAP@0.5, precision, recall, F1.
Uses Ultralytics built-in validation + custom per-image analysis.
"""

import json
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

from config import MODELS_DIR, MODEL_VARIANTS


def evaluate_detection(data_yaml: str, model_variant: str = "yolov8s") -> dict:
    """
    Run YOLOv8 validation on test set.
    Returns dict with mAP, precision, recall, F1.
    """
    model_path = MODELS_DIR / MODEL_VARIANTS[model_variant]
    model = YOLO(str(model_path))

    results = model.val(data=data_yaml, split="test", verbose=False)

    metrics = {
        "model_variant": model_variant,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": 2 * float(results.box.mp) * float(results.box.mr)
             / max(float(results.box.mp) + float(results.box.mr), 1e-6),
    }

    return metrics


def compare_variants(data_yaml: str) -> list[dict]:
    """Evaluate all model variants and return comparison."""
    results = []
    for variant in MODEL_VARIANTS:
        model_path = MODELS_DIR / MODEL_VARIANTS[variant]
        if model_path.exists():
            metrics = evaluate_detection(data_yaml, variant)
            results.append(metrics)
            print(f"{variant}: mAP@0.5={metrics['mAP50']:.4f}, "
                  f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}")
        else:
            print(f"{variant}: model file not found, skipping")
    return results


def plot_comparison(results: list[dict], save_path: str | None = None):
    """Plot bar chart comparing model variants."""
    if not results:
        print("No results to plot.")
        return

    variants = [r["model_variant"] for r in results]
    metrics_names = ["mAP50", "precision", "recall", "f1"]
    x = np.arange(len(variants))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics_names):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys
    data_yaml = sys.argv[1] if len(sys.argv) > 1 else "data/data.yaml"
    results = compare_variants(data_yaml)
    plot_comparison(results, "evaluation/detection_comparison.png")
