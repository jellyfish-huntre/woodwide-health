"""
Isolation Forest Detection CLI - Classic ML Baseline

Demonstrates Isolation Forest anomaly detection as a more sophisticated baseline
compared to simple HR thresholding. Shows that even classic ML struggles with
the context problem.

Usage:
    python isolation_forest_detection.py <subject_id> [--contamination 0.1]
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from src.detectors.isolation_forest_detector import (
    IsolationForestDetector,
    analyze_performance
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Isolation Forest anomaly detection on preprocessed PPG-DaLiA data"
    )
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID to process (e.g., 1, 2, 3)"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of anomalies (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detection results to disk"
    )

    args = parser.parse_args()

    print("="*80)
    print("Isolation Forest Anomaly Detection (Classic ML Baseline)")
    print("="*80)
    print(f"Subject: {args.subject_id}")
    print(f"Contamination: {args.contamination:.1%}")
    print()

    # Load preprocessed data
    data_file = Path(f"data/processed/subject_{args.subject_id:02d}_processed.pkl")
    if not data_file.exists():
        print(f"Error: Preprocessed data not found at {data_file}")
        print("Run preprocessing first:")
        print(f"  python -m src.ingestion.preprocess")
        return 1

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']

    print(f"Loaded {len(windows)} windows")
    print(f"Window shape: {windows.shape}")
    print()

    # Initialize detector
    detector = IsolationForestDetector(contamination=args.contamination)

    # Fit on exercise data
    print("Training on exercise windows...")
    detector.fit(windows, labels, exercise_labels=[2, 3, 4])

    # Predict
    print("Running detection...")
    result = detector.predict(windows)

    # Analyze performance
    activity_map = {
        1: 'Sitting',
        2: 'Cycling',
        3: 'Walking',
        4: 'Ascending stairs',
        5: 'Descending stairs'
    }

    metrics = analyze_performance(result, labels, activity_map)

    # Display results
    print()
    print("="*80)
    print("DETECTION RESULTS")
    print("="*80)

    print(f"\nOverall:")
    print(f"  Total windows: {metrics['total_windows']}")
    print(f"  Alerts: {metrics['total_alerts']} ({metrics['alert_rate']:.1%})")

    print(f"\nExercise Windows:")
    print(f"  Total: {metrics['exercise']['total']}")
    print(f"  Alerts: {metrics['exercise']['alerts']} ({metrics['exercise']['rate']:.1%})")

    print(f"\nRest Windows:")
    print(f"  Total: {metrics['rest']['total']}")
    print(f"  Alerts: {metrics['rest']['alerts']} ({metrics['rest']['rate']:.1%})")

    print(f"\nBy Activity:")
    for activity, stats in metrics['by_activity'].items():
        print(f"  {activity:20s}: {stats['alerts']:3d}/{stats['total']:3d} ({stats['rate']:6.1%})")

    # Analysis
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)

    exercise_fp_rate = metrics['exercise']['rate']

    print(f"\n🔍 Key Insight:")
    print(f"   Isolation Forest alerts on {exercise_fp_rate:.1%} of exercise windows")
    print(f"   (False positives due to high HR during normal exercise)")
    print()
    print(f"   This is better than naive thresholding but still suffers from")
    print(f"   the context problem: it can't distinguish 'high HR during exercise'")
    print(f"   from 'high HR during rest'.")
    print()
    print(f"   Wood Wide solves this by learning signal relationships through")
    print(f"   embeddings, achieving ~5% false positive rate on exercise.")

    # Save results
    if args.save_results:
        output_dir = Path("data/isolation_forest_detection")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"subject_{args.subject_id:02d}_results.pkl"

        results_data = {
            'subject_id': args.subject_id,
            'contamination': args.contamination,
            'result': result,
            'metrics': metrics,
            'timestamps': timestamps,
            'labels': labels
        }

        with open(output_file, 'wb') as f:
            pickle.dump(results_data, f)

        print(f"\n✓ Results saved to {output_file}")

    print()
    print("="*80)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
