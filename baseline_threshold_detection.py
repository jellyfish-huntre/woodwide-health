"""
Baseline Threshold Detection Script

This script demonstrates the traditional approach to health monitoring using simple
heart rate thresholds. It serves as a baseline to compare against the embedding-based
approach that understands context.

The problem with threshold-based detection:
- High HR during exercise → False alarm
- High HR during sleep → True alarm
- No way to distinguish context

Usage:
    python baseline_threshold_detection.py 1
    python baseline_threshold_detection.py 1 --threshold 120
    python baseline_threshold_detection.py 1 --show-plot
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


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


def extract_heart_rate(windows: np.ndarray) -> np.ndarray:
    """Extract heart rate from windows using peak detection.

    Args:
        windows: Shape (n_windows, window_length, n_features)
                 Features: [PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG]

    Returns:
        Heart rate per window (mean HR over 30s window)
        Shape: (n_windows,)
    """
    from scipy.signal import find_peaks

    # PPG is first feature (index 0)
    ppg_windows = windows[:, :, 0]

    # Sampling rate is 32 Hz after preprocessing
    sampling_rate = 32.0

    hr_bpm = np.zeros(len(ppg_windows))

    for i, ppg_window in enumerate(ppg_windows):
        # Find peaks in PPG signal
        # Minimum distance between peaks: 0.4s (150 BPM max)
        peaks, _ = find_peaks(ppg_window, distance=int(sampling_rate * 0.4))

        if len(peaks) > 1:
            # Calculate heart rate from peak intervals
            peak_intervals = np.diff(peaks) / sampling_rate  # Intervals in seconds
            mean_interval = peak_intervals.mean()
            hr = 60.0 / mean_interval  # Convert to BPM
            hr_bpm[i] = hr
        else:
            # Not enough peaks, estimate from signal variability
            # Higher variability usually indicates higher HR
            hr_bpm[i] = 70.0 + ppg_window.std() * 30

    return hr_bpm


def apply_threshold_detection(
    hr_bpm: np.ndarray,
    threshold: float = 120.0
) -> np.ndarray:
    """Apply simple threshold detection.

    Args:
        hr_bpm: Heart rate in BPM, shape (n_windows,)
        threshold: HR threshold in BPM

    Returns:
        Boolean array indicating alerts (True = alert)
        Shape: (n_windows,)
    """
    return hr_bpm > threshold


def analyze_false_positives(
    alerts: np.ndarray,
    labels: np.ndarray,
    activity_map: Dict[int, str]
) -> pd.DataFrame:
    """Analyze false positive rate by activity.

    Args:
        alerts: Boolean array of alerts
        labels: Activity labels per window
        activity_map: Mapping from label to activity name

    Returns:
        DataFrame with false positive analysis
    """
    results = []

    for label, activity in activity_map.items():
        # Find windows with this activity
        activity_mask = labels == label
        n_windows = activity_mask.sum()

        if n_windows == 0:
            continue

        # Count alerts during this activity
        alerts_during_activity = (alerts & activity_mask).sum()
        alert_rate = alerts_during_activity / n_windows * 100

        # Classify as exercise vs. rest
        exercise_activities = ['Cycling', 'Walking', 'Ascending stairs', 'Descending stairs']
        is_exercise = activity in exercise_activities

        results.append({
            'activity': activity,
            'type': 'Exercise' if is_exercise else 'Rest',
            'n_windows': n_windows,
            'n_alerts': alerts_during_activity,
            'alert_rate_pct': alert_rate
        })

    return pd.DataFrame(results)


def compute_detection_metrics(
    alerts: np.ndarray,
    labels: np.ndarray,
    exercise_labels: List[int]
) -> Dict:
    """Compute detection metrics.

    Args:
        alerts: Boolean array of alerts
        labels: Activity labels
        exercise_labels: List of label values that represent exercise

    Returns:
        Dictionary with metrics
    """
    # Create exercise mask
    is_exercise = np.isin(labels, exercise_labels)
    is_rest = ~is_exercise

    # Count alerts
    total_alerts = alerts.sum()
    alerts_during_exercise = (alerts & is_exercise).sum()
    alerts_during_rest = (alerts & is_rest).sum()

    # False positive rate (alerts during exercise)
    false_positive_rate = alerts_during_exercise / is_exercise.sum() * 100 if is_exercise.sum() > 0 else 0

    # True positive rate (alerts during rest - assuming high HR during rest is concerning)
    true_positive_rate = alerts_during_rest / is_rest.sum() * 100 if is_rest.sum() > 0 else 0

    return {
        'total_windows': len(alerts),
        'total_alerts': int(total_alerts),
        'alerts_during_exercise': int(alerts_during_exercise),
        'alerts_during_rest': int(alerts_during_rest),
        'false_positive_rate_pct': false_positive_rate,
        'true_positive_rate_pct': true_positive_rate,
        'exercise_windows': int(is_exercise.sum()),
        'rest_windows': int(is_rest.sum())
    }


def plot_detection_results(
    hr_bpm: np.ndarray,
    alerts: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    activity_map: Dict[int, str],
    save_path: str = None
):
    """Plot threshold detection results.

    Args:
        hr_bpm: Heart rate values
        alerts: Alert boolean array
        labels: Activity labels
        timestamps: Window timestamps
        threshold: HR threshold used
        activity_map: Activity name mapping
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Convert timestamps to minutes
    time_minutes = (timestamps - timestamps[0]) / 60

    # Plot 1: Heart rate over time
    ax1 = axes[0]
    ax1.plot(time_minutes, hr_bpm, 'b-', linewidth=1.5, label='Heart Rate')
    ax1.axhline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold} BPM)')
    ax1.scatter(time_minutes[alerts], hr_bpm[alerts], color='red', s=100,
                marker='x', linewidth=3, label='Alerts', zorder=5)
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax1.set_title('Threshold-Based Detection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Activity labels
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(activity_map)))
    for i, (label, activity) in enumerate(activity_map.items()):
        mask = labels == label
        if mask.any():
            ax2.scatter(time_minutes[mask], np.ones(mask.sum()) * label,
                       color=colors[i], label=activity, s=50)
    ax2.set_ylabel('Activity', fontsize=12)
    ax2.set_yticks(list(activity_map.keys()))
    ax2.set_yticklabels([activity_map[i] for i in activity_map.keys()], fontsize=9)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Alert timeline
    ax3 = axes[2]
    alert_values = alerts.astype(int)
    ax3.fill_between(time_minutes, 0, alert_values, color='red', alpha=0.3, label='Alert Active')
    ax3.set_xlabel('Time (minutes)', fontsize=12)
    ax3.set_ylabel('Alert Status', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'Alert'])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Baseline threshold-based heart rate detection"
    )
    parser.add_argument(
        'subject_id',
        type=int,
        help='Subject ID (1-15)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=120.0,
        help='Heart rate threshold in BPM (default: 120)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with preprocessed data'
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

    args = parser.parse_args()

    print("=" * 80)
    print("BASELINE THRESHOLD DETECTION")
    print("=" * 80)
    print(f"\nSubject: {args.subject_id}")
    print(f"Threshold: {args.threshold} BPM")
    print()

    # Load data
    print("Loading preprocessed data...")
    data = load_preprocessed_data(args.subject_id, args.data_dir)

    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']

    # Activity mapping (from PPG-DaLiA dataset)
    activity_map = {
        1: 'Sitting',
        2: 'Cycling',
        3: 'Walking',
        4: 'Ascending stairs',
        5: 'Descending stairs',
        6: 'Vacuum cleaning',
        7: 'Ironing'
    }

    # Exercise vs. rest classification
    exercise_labels = [2, 3, 4, 7]  # Cycling, Walking, Stairs

    print(f"✓ Loaded {len(windows)} windows")
    print(f"  Window shape: {windows.shape}")
    print(f"  Duration: {(timestamps[-1] - timestamps[0]) / 60:.1f} minutes")
    print()

    # Extract heart rate
    print("Extracting heart rate from PPG signal...")
    hr_bpm = extract_heart_rate(windows)
    print(f"✓ Heart rate range: {hr_bpm.min():.1f} - {hr_bpm.max():.1f} BPM")
    print(f"  Mean: {hr_bpm.mean():.1f} BPM")
    print()

    # Apply threshold detection
    print(f"Applying threshold detection (HR > {args.threshold} BPM)...")
    alerts = apply_threshold_detection(hr_bpm, threshold=args.threshold)
    print(f"✓ Generated {alerts.sum()} alerts")
    print()

    # Analyze false positives
    print("Analyzing detection performance by activity...")
    fp_analysis = analyze_false_positives(alerts, labels, activity_map)
    print()
    print(fp_analysis.to_string(index=False))
    print()

    # Compute overall metrics
    print("Overall Detection Metrics:")
    print("-" * 80)
    metrics = compute_detection_metrics(alerts, labels, exercise_labels)

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
    print("THE PROBLEM WITH THRESHOLD-BASED DETECTION:")
    print("=" * 80)
    if metrics['alerts_during_exercise'] > 0:
        print(f"⚠️  {metrics['alerts_during_exercise']} FALSE ALARMS during exercise")
        print(f"   ({metrics['false_positive_rate_pct']:.1f}% of exercise windows)")
        print()
        print("   Traditional fitness trackers can't distinguish:")
        print("   • High HR during exercise (normal) ✓")
        print("   • High HR during rest (concerning) ⚠️")
        print()
        print("   This is the 'context problem' that embeddings solve!")
    else:
        print("✓ No false alarms (threshold may be too high)")
    print("=" * 80)
    print()

    # Save results
    output_dir = Path('data/baseline_detection')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'subject_{args.subject_id:02d}_threshold_{int(args.threshold)}.pkl'
    results = {
        'subject_id': args.subject_id,
        'threshold': args.threshold,
        'hr_bpm': hr_bpm,
        'alerts': alerts,
        'metrics': metrics,
        'fp_analysis': fp_analysis
    }

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {results_file}")
    print()

    # Plot results
    if args.show_plot or args.save_plot:
        print("Generating visualization...")
        plot_path = args.save_plot if args.save_plot else None
        if not plot_path and args.show_plot:
            plot_detection_results(
                hr_bpm, alerts, labels, timestamps,
                args.threshold, activity_map
            )
        else:
            plot_detection_results(
                hr_bpm, alerts, labels, timestamps,
                args.threshold, activity_map, save_path=plot_path
            )


if __name__ == "__main__":
    main()
