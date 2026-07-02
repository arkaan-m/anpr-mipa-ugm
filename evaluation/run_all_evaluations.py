"""
Run all evaluations and generate a combined report.
Usage: python -m evaluation.run_all_evaluations --data data/data.yaml --test-dir data/processed/images/test
"""

import argparse
import json
import glob
import time
from pathlib import Path

from evaluation.detection_metrics import compare_variants, plot_comparison
from evaluation.ocr_metrics import load_test_data, evaluate_ocr, plot_ocr_comparison
from evaluation.ablation_study import run_ablation, plot_ablation
from evaluation.timing_benchmark import benchmark_batch, plot_timing


def main():
    parser = argparse.ArgumentParser(description="Run all ANPR evaluations")
    parser.add_argument("--data", default="data/data.yaml", help="YOLO data.yaml path")
    parser.add_argument("--test-dir", default="data/processed/images/test", help="Test image directory")
    parser.add_argument("--ocr-csv", default="data/test_ocr.csv", help="OCR test CSV path")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory")
    parser.add_argument("--model", default="yolov8s", help="Model variant for ablation/timing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted(
        glob.glob(f"{args.test_dir}/*.jpg") + glob.glob(f"{args.test_dir}/*.png")
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S"),
        "model": args.model,
        "test_images": len(test_images),
    }

    # 1. Detection metrics
    print("\n" + "="*60)
    print("1/4  DETECTION METRICS")
    print("="*60)
    detection_results = compare_variants(args.data)
    plot_comparison(detection_results, str(output_dir / "detection_comparison.png"))
    report["detection"] = detection_results

    # 2. OCR metrics
    if Path(args.ocr_csv).exists():
        print("\n" + "="*60)
        print("2/4  OCR METRICS")
        print("="*60)
        test_data = load_test_data(args.ocr_csv)
        ocr_with = evaluate_ocr(test_data, enable_preprocessing=True)
        ocr_without = evaluate_ocr(test_data, enable_preprocessing=False)
        plot_ocr_comparison(ocr_with, str(output_dir / "ocr_comparison.png"))
        report["ocr_with_preprocessing"] = ocr_with
        report["ocr_without_preprocessing"] = ocr_without
    else:
        print(f"\n[SKIP] OCR CSV not found at {args.ocr_csv}")

    # 3. Ablation study
    if test_images:
        print("\n" + "="*60)
        print("3/4  ABLATION STUDY")
        print("="*60)
        ablation_results = run_ablation(test_images[:50], args.model)
        plot_ablation(ablation_results, str(output_dir / "ablation_results.png"))
        # Serialize (remove non-serializable fields)
        report["ablation"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "plates_detected"}
            for k, v in ablation_results.items()
        }

        # 4. Timing benchmark
        print("\n" + "="*60)
        print("4/4  TIMING BENCHMARK")
        print("="*60)
        timing_summary = benchmark_batch(test_images[:20], args.model)
        plot_timing(timing_summary, str(output_dir / "timing_benchmark.png"))
        report["timing"] = timing_summary
    else:
        print(f"\n[SKIP] No test images found in {args.test_dir}")

    # Save combined report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
