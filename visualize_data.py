"""
Visualization utility for inspecting preprocessed data.

Displays PPG and accelerometer signals to verify preprocessing quality
before sending to Wood Wide API for embedding generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse


def load_processed_subject(subject_id: int, data_dir: str = "data/processed"):
    """
    Load preprocessed subject data.

    Args:
        subject_id: Subject ID (1-15)
        data_dir: Directory containing processed data

    Returns:
        Processed data dictionary
    """
    file_path = Path(data_dir) / f"subject_{subject_id:02d}_processed.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def plot_signal_overview(data: dict, window_idx: int = 0):
    """
    Plot overview of signals for a specific window.

    Args:
        data: Processed data dictionary
        window_idx: Index of window to visualize
    """
    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']
    metadata = data['metadata']

    if window_idx >= len(windows):
        raise ValueError(f"Window index {window_idx} out of range (max: {len(windows)-1})")

    window = windows[window_idx]  # Shape: (window_length, n_features)
    window_time = timestamps[window_idx]
    activity = labels[window_idx]

    # Feature names (from preprocessor)
    feature_names = ['PPG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_MAG']

    # Create time axis for this window
    sampling_rate = metadata['sampling_rate']
    window_seconds = metadata['window_seconds']
    time_axis = np.linspace(0, window_seconds, window.shape[0])

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Subject {data['subject_id']} - Window {window_idx} "
        f"(t={window_time:.1f}s, Activity={activity})",
        fontsize=14,
        fontweight='bold'
    )

    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        signal = window[:, i]

        ax.plot(time_axis, signal, linewidth=1.5)
        ax.set_ylabel(feature_name, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        ax.text(
            0.02, 0.95,
            f'μ={mean_val:.3f}, σ={std_val:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    axes[-1].set_xlabel('Time (seconds)', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_activity_distribution(data: dict):
    """
    Plot distribution of activities across all windows.

    Args:
        data: Processed data dictionary
    """
    labels = data['labels']
    unique_activities, counts = np.unique(labels, return_counts=True)

    # Activity names (PPG-DaLiA activity codes)
    activity_names = {
        0: 'Transient',
        1: 'Sitting',
        2: 'Walking',
        3: 'Cycling',
        4: 'Driving',
        5: 'Lunch',
        6: 'Stairs',
        7: 'Table Soccer'
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        unique_activities,
        counts,
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )

    # Add labels
    ax.set_xlabel('Activity Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Windows', fontweight='bold', fontsize=12)
    ax.set_title(
        f"Subject {data['subject_id']} - Activity Distribution",
        fontweight='bold',
        fontsize=14
    )
    ax.set_xticks(unique_activities)
    ax.set_xticklabels(
        [activity_names.get(a, f'Unknown ({a})') for a in unique_activities],
        rotation=45,
        ha='right'
    )
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    return fig


def plot_feature_correlation(data: dict, sample_windows: int = 100):
    """
    Plot correlation matrix between features.

    Args:
        data: Processed data dictionary
        sample_windows: Number of windows to sample for correlation
    """
    windows = data['windows']

    # Sample random windows
    n_windows = min(sample_windows, len(windows))
    sampled_indices = np.random.choice(len(windows), n_windows, replace=False)
    sampled_windows = windows[sampled_indices]

    # Flatten windows: (n_windows, window_length, n_features) -> (n_samples, n_features)
    flattened = sampled_windows.reshape(-1, sampled_windows.shape[2])

    # Compute correlation
    correlation = np.corrcoef(flattened.T)

    feature_names = ['PPG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_MAG']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(
                j, i, f'{correlation[i, j]:.2f}',
                ha="center", va="center",
                color="black" if abs(correlation[i, j]) < 0.5 else "white",
                fontweight='bold'
            )

    ax.set_title(
        f"Subject {data['subject_id']} - Feature Correlation Matrix",
        fontweight='bold',
        fontsize=14
    )
    fig.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    return fig


def main():
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed PPG-DaLiA data"
    )
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID to visualize (1-15)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window index to visualize (default: 0)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed data (default: data/processed)"
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading subject {args.subject_id}...")
        data = load_processed_subject(args.subject_id, args.data_dir)

        print(f"Subject {data['subject_id']} loaded successfully!")
        print(f"  Windows: {data['metadata']['n_windows']}")
        print(f"  Window shape: {data['metadata']['window_shape']}")
        print(f"  Window duration: {data['metadata']['window_seconds']}s")

        # Create visualizations
        print(f"\nGenerating visualizations...")

        # 1. Signal overview
        fig1 = plot_signal_overview(data, window_idx=args.window)

        # 2. Activity distribution
        fig2 = plot_activity_distribution(data)

        # 3. Feature correlation
        fig3 = plot_feature_correlation(data)

        plt.show()

        print("✓ Visualization complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_all_subjects.py first to preprocess the data.")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
