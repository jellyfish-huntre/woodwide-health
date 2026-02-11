"""
Three-Way Comparison Visualization

Standalone script to visualize Baseline, Isolation Forest, and Wood Wide
detection results side-by-side.

Usage:
    python three_way_visualization.py <subject_id>
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_helpers import (
    compute_isolation_forest_metrics,
    create_three_way_comparison_chart,
    create_three_way_timeline,
    create_comparison_table
)


def extract_heart_rate_simple(windows: np.ndarray) -> np.ndarray:
    """Quick HR extraction for visualization."""
    from scipy.signal import find_peaks

    ppg_windows = windows[:, :, 0]
    sampling_rate = 32.0
    hr_bpm = np.zeros(len(ppg_windows))

    for i, ppg_window in enumerate(ppg_windows):
        peaks, _ = find_peaks(ppg_window, distance=int(sampling_rate * 0.4))
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sampling_rate
            hr_bpm[i] = 60.0 / peak_intervals.mean()
        else:
            hr_bpm[i] = 70.0

    return hr_bpm


def main():
    parser = argparse.ArgumentParser(description="Three-way detection comparison")
    parser.add_argument("subject_id", type=int, help="Subject ID (1, 2, or 3)")
    parser.add_argument("--baseline-threshold", type=int, default=100, help="HR threshold for baseline")
    parser.add_argument("--save-html", action="store_true", help="Save as interactive HTML")

    args = parser.parse_args()

    print("="*80)
    print(f"Three-Way Detection Comparison - Subject {args.subject_id}")
    print("="*80)

    # Load preprocessed data
    data_file = Path(f"data/processed/subject_{args.subject_id:02d}_processed.pkl")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return 1

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']

    # Extract HR
    hr_bpm = extract_heart_rate_simple(windows)

    # Method 1: Baseline Threshold
    baseline_alerts = hr_bpm > args.baseline_threshold
    is_exercise = np.isin(labels, [2, 3, 4, 5])
    baseline_metrics = {
        'total_alerts': int(baseline_alerts.sum()),
        'alerts_during_exercise': int((baseline_alerts & is_exercise).sum()),
        'alerts_during_rest': int((baseline_alerts & ~is_exercise).sum()),
        'false_positive_rate_pct': (
            (baseline_alerts & is_exercise).sum() / is_exercise.sum() * 100
            if is_exercise.sum() > 0 else 0
        ),
        'exercise_windows': int(is_exercise.sum()),
        'rest_windows': int((~is_exercise).sum())
    }

    # Method 2: Isolation Forest
    if_file = Path(f"data/isolation_forest_detection/subject_{args.subject_id:02d}_results.pkl")
    if not if_file.exists():
        print(f"\nWarning: Isolation Forest results not found.")
        print(f"Run: python isolation_forest_detection.py {args.subject_id} --save-results")
        return 1

    with open(if_file, 'rb') as f:
        if_data = pickle.load(f)

    if_alerts = if_data['result'].alerts
    if_metrics = compute_isolation_forest_metrics(if_alerts, labels)

    # Method 3: Wood Wide
    ww_file = Path(f"data/woodwide_detection/subject_{args.subject_id:02d}_results.pkl")
    if not ww_file.exists():
        print(f"\nWarning: Wood Wide results not found.")
        print(f"Run: python woodwide_detection.py {args.subject_id} --use-mock")
        return 1

    with open(ww_file, 'rb') as f:
        ww_data = pickle.load(f)

    ww_alerts = ww_data['result'].alerts
    ww_metrics = ww_data['result'].metrics

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print("\nExercise False Positive Rates:")
    print(f"  Baseline Threshold:  {baseline_metrics['false_positive_rate_pct']:6.1f}%")
    print(f"  Isolation Forest:    {if_metrics['false_positive_rate_pct']:6.1f}%  (↓{baseline_metrics['false_positive_rate_pct'] - if_metrics['false_positive_rate_pct']:.1f}%)")
    print(f"  Wood Wide:           {ww_metrics['false_positive_rate_pct']:6.1f}%  (↓{baseline_metrics['false_positive_rate_pct'] - ww_metrics['false_positive_rate_pct']:.1f}%)")

    if baseline_metrics['false_positive_rate_pct'] > 0:
        if_improvement = (1 - if_metrics['false_positive_rate_pct'] / baseline_metrics['false_positive_rate_pct']) * 100
        ww_improvement = (1 - ww_metrics['false_positive_rate_pct'] / baseline_metrics['false_positive_rate_pct']) * 100
        print(f"\nImprovement vs Baseline:")
        print(f"  Isolation Forest:    {if_improvement:5.1f}% better")
        print(f"  Wood Wide:           {ww_improvement:5.1f}% better")

    print("\nTotal Alerts:")
    print(f"  Baseline:            {baseline_metrics['total_alerts']}")
    print(f"  Isolation Forest:    {if_metrics['total_alerts']}")
    print(f"  Wood Wide:           {ww_metrics['total_alerts']}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. FP Rate comparison chart
    fp_chart = create_three_way_comparison_chart(
        baseline_metrics['false_positive_rate_pct'],
        if_metrics['false_positive_rate_pct'],
        ww_metrics['false_positive_rate_pct']
    )

    # 2. Timeline comparison
    timeline = create_three_way_timeline(
        timestamps,
        hr_bpm,
        baseline_alerts,
        if_alerts,
        ww_alerts
    )

    # 3. Create combined figure
    from plotly.subplots import make_subplots

    combined_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            'Exercise False Positive Rate Comparison',
            'Alert Timeline Comparison'
        ),
        vertical_spacing=0.15,
        row_heights=[0.3, 0.7],
        specs=[[{"type": "bar"}], [{"type": "scatter"}]]
    )

    # Add FP chart to top
    for trace in fp_chart.data:
        combined_fig.add_trace(trace, row=1, col=1)

    # Update axis from FP chart
    combined_fig.update_yaxes(
        title_text="False Positive Rate (%)",
        row=1, col=1,
        range=[0, min(baseline_metrics['false_positive_rate_pct'] * 1.2, 105)]
    )

    # Show in browser
    combined_fig.update_layout(height=900, showlegend=False, title_text=f"Subject {args.subject_id}: Three-Way Detection Comparison")

    if args.save_html:
        output_file = f"three_way_comparison_subject_{args.subject_id}.html"
        combined_fig.write_html(output_file)
        print(f"\n✓ Saved to {output_file}")
    else:
        combined_fig.show()
        print("\n✓ Visualization opened in browser")

    # Print table
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80)

    table = create_comparison_table(baseline_metrics, if_metrics, ww_metrics)
    print(table.to_string(index=False))

    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
