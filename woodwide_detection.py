"""
Wood Wide Embedding-Based Signal Decoupling Detection

This script demonstrates the Wood Wide approach to health monitoring using
multivariate embeddings to understand context and detect signal decoupling.

The Wood Wide detector:
- Learns what "normal" looks like during exercise (high HR + high activity)
- Detects anomalies when embeddings deviate from normal patterns
- Achieves low false positive rates by understanding context

Usage:
    python woodwide_detection.py 1
    python woodwide_detection.py 1 --threshold-percentile 95
    python woodwide_detection.py 1 --show-plot
    python woodwide_detection.py 1 --use-mock  # Use mock API for testing
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from src.detectors.woodwide import WoodWideDetector, DetectionResult
from src.embeddings.generate import send_windows_to_woodwide, load_embeddings


def load_preprocessed_data(subject_id: int, data_dir: str = "data/processed") -> Dict:
    """Load preprocessed subject data.

    Args:
        subject_id: Subject ID (1-15)
        data_dir: Directory containing preprocessed data

    Returns:
        Dictionary containing windows, timestamps, labels, metadata

    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
    """
    data_path = Path(data_dir) / f"subject_{subject_id:02d}_processed.pkl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}\n"
            f"Run preprocessing first: python -m src.ingestion.preprocess"
        )

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_or_generate_embeddings(
    subject_id: int,
    windows: np.ndarray,
    use_mock: bool = True,
    force_regenerate: bool = False
) -> Tuple[np.ndarray, Dict]:
    """Get embeddings (load from cache or generate new).

    Args:
        subject_id: Subject ID
        windows: Preprocessed windows
        use_mock: Use mock API instead of real API
        force_regenerate: Force regeneration even if cached

    Returns:
        Tuple of (embeddings, metadata)
    """
    embeddings_dir = Path("data/embeddings")
    embeddings_file = embeddings_dir / f"subject_{subject_id:02d}_embeddings.npy"
    metadata_file = embeddings_dir / f"subject_{subject_id:02d}_metadata.pkl"

    # Try to load cached embeddings
    if not force_regenerate and embeddings_file.exists() and metadata_file.exists():
        print(f"Loading cached embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        return embeddings, metadata

    # Generate new embeddings
    print(f"Generating embeddings using {'mock' if use_mock else 'real'} API...")
    embeddings, metadata = send_windows_to_woodwide(
        windows,
        batch_size=32,
        embedding_dim=128,
        use_mock=use_mock
    )

    # Save for future use
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ Saved embeddings to {embeddings_file}")

    return embeddings, metadata


def analyze_performance_by_activity(
    result: DetectionResult,
    labels: np.ndarray,
    activity_map: Dict[int, str]
) -> pd.DataFrame:
    """Analyze detection performance by activity.

    Args:
        result: Detection result
        labels: Activity labels
        activity_map: Mapping from label to activity name

    Returns:
        DataFrame with per-activity analysis
    """
    analysis = []

    for label, activity in activity_map.items():
        activity_mask = labels == label
        n_windows = activity_mask.sum()

        if n_windows == 0:
            continue

        # Count alerts during this activity
        alerts_during_activity = (result.alerts & activity_mask).sum()
        alert_rate = alerts_during_activity / n_windows * 100

        # Compute mean distance for this activity
        mean_distance = result.distances[activity_mask].mean()
        max_distance = result.distances[activity_mask].max()

        # Classify activity type
        exercise_activities = ['Cycling', 'Walking', 'Ascending stairs', 'Descending stairs']
        is_exercise = activity in exercise_activities

        analysis.append({
            'activity': activity,
            'type': 'Exercise' if is_exercise else 'Rest',
            'n_windows': n_windows,
            'n_alerts': alerts_during_activity,
            'alert_rate_pct': alert_rate,
            'mean_distance': mean_distance,
            'max_distance': max_distance
        })

    return pd.DataFrame(analysis)


def plot_woodwide_detection(
    result: DetectionResult,
    labels: np.ndarray,
    timestamps: np.ndarray,
    activity_map: Dict[int, str],
    save_path: str = None
):
    """Plot Wood Wide detection results.

    Args:
        result: Detection result
        labels: Activity labels
        timestamps: Window timestamps
        activity_map: Activity name mapping
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Convert timestamps to minutes
    time_minutes = (timestamps - timestamps[0]) / 60

    # Plot 1: Distances from normal centroid
    ax1 = axes[0]
    ax1.plot(time_minutes, result.distances, 'b-', linewidth=1.5, label='Distance from Normal')
    ax1.axhline(result.threshold, color='r', linestyle='--', linewidth=2,
                label=f'Threshold ({result.threshold:.3f})')
    ax1.scatter(time_minutes[result.alerts], result.distances[result.alerts],
                color='red', s=100, marker='x', linewidth=3, label='Alerts', zorder=5)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.set_title('Wood Wide Embedding-Based Detection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance distribution by activity
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(activity_map)))
    for i, (label, activity) in enumerate(activity_map.items()):
        mask = labels == label
        if mask.any():
            ax2.scatter(time_minutes[mask], result.distances[mask],
                       color=colors[i], label=activity, s=30, alpha=0.6)
    ax2.axhline(result.threshold, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Distance', fontsize=12)
    ax2.set_title('Distance by Activity Type', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Activity labels
    ax3 = axes[2]
    for i, (label, activity) in enumerate(activity_map.items()):
        mask = labels == label
        if mask.any():
            ax3.scatter(time_minutes[mask], np.ones(mask.sum()) * label,
                       color=colors[i], label=activity, s=50)
    ax3.set_ylabel('Activity', fontsize=12)
    ax3.set_yticks(list(activity_map.keys()))
    ax3.set_yticklabels([activity_map[i] for i in activity_map.keys()], fontsize=9)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Alert timeline
    ax4 = axes[3]
    alert_values = result.alerts.astype(int)
    ax4.fill_between(time_minutes, 0, alert_values, color='red', alpha=0.3, label='Alert Active')
    ax4.set_xlabel('Time (minutes)', fontsize=12)
    ax4.set_ylabel('Alert Status', fontsize=12)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Normal', 'Alert'])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def compare_with_baseline(
    woodwide_metrics: Dict,
    subject_id: int,
    baseline_threshold: int = 100
):
    """Compare Wood Wide performance with baseline threshold detection.

    Args:
        woodwide_metrics: Metrics from Wood Wide detector
        subject_id: Subject ID
        baseline_threshold: HR threshold used in baseline
    """
    baseline_file = Path(f"data/baseline_detection/subject_{subject_id:02d}_threshold_{baseline_threshold}.pkl")

    if not baseline_file.exists():
        print(f"\n⚠️  Baseline results not found: {baseline_file}")
        print("   Run baseline detection first:")
        print(f"   python baseline_threshold_detection.py {subject_id} --threshold {baseline_threshold}")
        return

    # Load baseline results
    with open(baseline_file, 'rb') as f:
        baseline_results = pickle.load(f)

    baseline_metrics = baseline_results['metrics']

    # Create comparison table
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: Wood Wide vs. Baseline")
    print("=" * 80)
    print()

    comparison_data = [
        ['Metric', 'Baseline (Threshold)', 'Wood Wide (Embeddings)', 'Improvement'],
        ['-' * 30, '-' * 25, '-' * 25, '-' * 15],
        [
            'False Positive Rate',
            f"{baseline_metrics['false_positive_rate_pct']:.1f}%",
            f"{woodwide_metrics['false_positive_rate_pct']:.1f}%",
            f"{baseline_metrics['false_positive_rate_pct'] - woodwide_metrics['false_positive_rate_pct']:.1f}% ↓"
        ],
        [
            'Alerts During Exercise',
            f"{baseline_metrics['alerts_during_exercise']} / {baseline_metrics['exercise_windows']}",
            f"{woodwide_metrics['alerts_during_exercise']} / {woodwide_metrics['exercise_windows']}",
            f"{baseline_metrics['alerts_during_exercise'] - woodwide_metrics['alerts_during_exercise']} fewer"
        ],
        [
            'Alerts During Rest',
            f"{baseline_metrics['alerts_during_rest']} / {baseline_metrics['rest_windows']}",
            f"{woodwide_metrics['alerts_during_rest']} / {woodwide_metrics['rest_windows']}",
            '-'
        ],
        [
            'Total Alerts',
            f"{baseline_metrics['total_alerts']}",
            f"{woodwide_metrics['total_alerts']}",
            f"{baseline_metrics['total_alerts'] - woodwide_metrics['total_alerts']} fewer"
        ]
    ]

    for row in comparison_data:
        print(f"{row[0]:<30} {row[1]:<25} {row[2]:<25} {row[3]:<15}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Wood Wide embedding-based signal decoupling detection"
    )
    parser.add_argument(
        'subject_id',
        type=int,
        help='Subject ID (1-15)'
    )
    parser.add_argument(
        '--threshold-percentile',
        type=float,
        default=95.0,
        help='Distance threshold percentile (default: 95)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with preprocessed data'
    )
    parser.add_argument(
        '--use-mock',
        action='store_true',
        help='Use mock API for embeddings'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regenerate embeddings even if cached'
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display visualization plot'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save plot to file'
    )
    parser.add_argument(
        '--compare-baseline',
        type=int,
        default=100,
        help='Compare with baseline threshold detection (default: 100 BPM)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WOOD WIDE EMBEDDING-BASED DETECTION")
    print("=" * 80)
    print(f"\nSubject: {args.subject_id}")
    print(f"Threshold Percentile: {args.threshold_percentile}")
    print(f"Using: {'Mock' if args.use_mock else 'Real'} API")
    print()

    # Load preprocessed data
    print("Loading preprocessed data...")
    data = load_preprocessed_data(args.subject_id, args.data_dir)

    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']

    # Activity mapping
    activity_map = {
        1: 'Sitting',
        2: 'Cycling',
        3: 'Walking',
        4: 'Ascending stairs',
        5: 'Descending stairs',
        6: 'Vacuum cleaning',
        7: 'Ironing'
    }

    print(f"✓ Loaded {len(windows)} windows")
    print(f"  Window shape: {windows.shape}")
    print(f"  Duration: {(timestamps[-1] - timestamps[0]) / 60:.1f} minutes")
    print()

    # Get or generate embeddings
    embeddings, emb_metadata = get_or_generate_embeddings(
        args.subject_id,
        windows,
        use_mock=args.use_mock,
        force_regenerate=args.force_regenerate
    )

    print(f"✓ Embeddings shape: {embeddings.shape}")
    print()

    # Create and fit Wood Wide detector
    print("Fitting Wood Wide detector...")
    detector = WoodWideDetector(threshold_percentile=args.threshold_percentile)

    result = detector.fit_predict(embeddings, labels)

    print(f"✓ Learned normal activity centroid")
    print(f"  Distance threshold: {result.threshold:.4f}")
    print()

    # Analyze performance by activity
    print("Detection Performance by Activity:")
    print("-" * 80)
    activity_analysis = analyze_performance_by_activity(result, labels, activity_map)
    print()
    print(activity_analysis.to_string(index=False))
    print()

    # Overall metrics
    print("Overall Detection Metrics:")
    print("-" * 80)
    metrics = result.metrics

    print(f"Total Windows: {metrics['total_windows']}")
    print(f"Total Alerts: {metrics['total_alerts']} ({metrics['total_alerts']/metrics['total_windows']*100:.1f}%)")
    print()
    print(f"Exercise Windows: {metrics['exercise_windows']}")
    print(f"  Alerts during exercise: {metrics['alerts_during_exercise']} "
          f"({metrics['false_positive_rate_pct']:.1f}% false positive rate)")
    print()
    print(f"Rest Windows: {metrics['rest_windows']}")
    print(f"  Alerts during rest: {metrics['alerts_during_rest']} "
          f"({metrics['true_positive_rate_pct']:.1f}% detection rate)")
    print()

    # Summary
    print("=" * 80)
    print("WOOD WIDE EMBEDDING-BASED DETECTION RESULTS:")
    print("=" * 80)
    if metrics['false_positive_rate_pct'] < 20:
        print(f"✓ LOW false positive rate: {metrics['false_positive_rate_pct']:.1f}%")
        print()
        print("   The embedding-based approach successfully distinguishes:")
        print("   • High HR during exercise (normal) ✓")
        print("   • High HR during rest (concerning) ⚠️")
        print()
        print("   This solves the 'context problem'!")
    else:
        print(f"⚠️  False positive rate: {metrics['false_positive_rate_pct']:.1f}%")
        print("   (Consider adjusting threshold-percentile)")
    print("=" * 80)
    print()

    # Save results
    output_dir = Path('data/woodwide_detection')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'subject_{args.subject_id:02d}_results.pkl'
    save_data = {
        'subject_id': args.subject_id,
        'result': result,
        'embeddings': embeddings,
        'activity_analysis': activity_analysis
    }

    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"✓ Results saved to {results_file}")
    print()

    # Save detector
    detector_file = output_dir / f'subject_{args.subject_id:02d}_detector.pkl'
    detector.save(detector_file)
    print(f"✓ Detector saved to {detector_file}")
    print()

    # Compare with baseline
    if args.compare_baseline:
        compare_with_baseline(metrics, args.subject_id, args.compare_baseline)
        print()

    # Plot results
    if args.show_plot or args.save_plot:
        print("Generating visualization...")
        plot_path = args.save_plot if args.save_plot else None
        if not plot_path and args.show_plot:
            plot_woodwide_detection(result, labels, timestamps, activity_map)
        else:
            plot_woodwide_detection(result, labels, timestamps, activity_map, save_path=plot_path)


if __name__ == "__main__":
    main()
