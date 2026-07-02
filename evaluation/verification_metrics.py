"""
Verification evaluation: accuracy, FAR, FRR, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def evaluate_verification(predictions: list[dict]) -> dict:
    """
    Evaluate verification performance.

    predictions: list of {
        "predicted_status": "AUTHORIZED"|"UNAUTHORIZED"|"UNCERTAIN",
        "actual_status": "AUTHORIZED"|"UNAUTHORIZED",
        "plate_text": str,
    }

    Returns accuracy, FAR, FRR, and confusion matrix data.
    """
    # Filter out UNCERTAIN and OCR_FAILED for binary metrics
    binary_preds = [p for p in predictions if p["predicted_status"] in ("AUTHORIZED", "UNAUTHORIZED")]

    total = len(binary_preds)
    if total == 0:
        return {"accuracy": 0, "far": 0, "frr": 0, "total": 0}

    tp = sum(1 for p in binary_preds
             if p["predicted_status"] == "AUTHORIZED" and p["actual_status"] == "AUTHORIZED")
    tn = sum(1 for p in binary_preds
             if p["predicted_status"] == "UNAUTHORIZED" and p["actual_status"] == "UNAUTHORIZED")
    fp = sum(1 for p in binary_preds
             if p["predicted_status"] == "AUTHORIZED" and p["actual_status"] == "UNAUTHORIZED")
    fn = sum(1 for p in binary_preds
             if p["predicted_status"] == "UNAUTHORIZED" and p["actual_status"] == "AUTHORIZED")

    accuracy = (tp + tn) / total if total else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate

    # Full confusion matrix (including UNCERTAIN)
    labels = ["AUTHORIZED", "UNAUTHORIZED", "UNCERTAIN"]
    confusion = np.zeros((3, 3), dtype=int)

    label_to_idx = {l: i for i, l in enumerate(labels)}
    for p in predictions:
        pred_idx = label_to_idx.get(p["predicted_status"])
        actual_idx = label_to_idx.get(p["actual_status"])
        if pred_idx is not None and actual_idx is not None:
            confusion[actual_idx][pred_idx] += 1

    # Status distribution
    pred_counts = Counter(p["predicted_status"] for p in predictions)
    actual_counts = Counter(p["actual_status"] for p in predictions)

    return {
        "total": len(predictions),
        "binary_total": total,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": confusion,
        "labels": labels,
        "predicted_distribution": dict(pred_counts),
        "actual_distribution": dict(actual_counts),
    }


def plot_confusion_matrix(metrics: dict, save_path: str | None = None):
    """Plot verification confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["labels"],
        yticklabels=metrics["labels"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Verification Confusion Matrix\n"
                 f"Accuracy={metrics['accuracy']:.3f}, "
                 f"FAR={metrics['far']:.3f}, FRR={metrics['frr']:.3f}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_report(metrics: dict):
    """Print a formatted verification report."""
    print("=" * 50)
    print("VERIFICATION EVALUATION REPORT")
    print("=" * 50)
    print(f"Total predictions:  {metrics['total']}")
    print(f"Binary (excl. UNCERTAIN): {metrics['binary_total']}")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"FAR:       {metrics['far']:.4f} ({metrics['far']*100:.1f}%)")
    print(f"FRR:       {metrics['frr']:.4f} ({metrics['frr']*100:.1f}%)")
    print()
    print(f"TP={metrics['tp']}, TN={metrics['tn']}, "
          f"FP={metrics['fp']}, FN={metrics['fn']}")
    print()
    print("Predicted distribution:", metrics.get("predicted_distribution", {}))
    print("Actual distribution:   ", metrics.get("actual_distribution", {}))
    print("=" * 50)
