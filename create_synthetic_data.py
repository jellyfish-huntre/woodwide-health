"""
Create synthetic PPG-DaLiA-like data for testing preprocessing pipeline.

This generates realistic test data matching the PPG-DaLiA format.
"""

import numpy as np
import pickle
from pathlib import Path


def generate_synthetic_subject(subject_id: int, duration_minutes: int = 10):
    """
    Generate synthetic data for one subject.

    Args:
        subject_id: Subject ID (1-15)
        duration_minutes: Duration of recording in minutes

    Returns:
        Dictionary matching PPG-DaLiA format
    """
    # Sampling rates
    ppg_rate = 64  # Hz
    acc_rate = 32  # Hz

    # Number of samples
    n_samples_ppg = int(duration_minutes * 60 * ppg_rate)
    n_samples_acc = int(duration_minutes * 60 * acc_rate)

    # Generate PPG signal (simulated heart rate ~60-120 bpm)
    t_ppg = np.arange(n_samples_ppg) / ppg_rate
    heart_rate = 70 + 20 * np.sin(2 * np.pi * 0.01 * t_ppg)  # Varying HR
    ppg_signal = np.sin(2 * np.pi * heart_rate / 60 * t_ppg)
    ppg_signal += 0.1 * np.random.randn(n_samples_ppg)  # Add noise

    # Generate accelerometer data (3-axis)
    t_acc = np.arange(n_samples_acc) / acc_rate

    # Simulate different activities over time
    activity_changes = [0, 2, 4, 6, 8]  # Minutes when activity changes
    activities = [1, 2, 3, 2, 1]  # sitting, walking, cycling, walking, sitting

    acc_x = np.zeros(n_samples_acc)
    acc_y = np.zeros(n_samples_acc)
    acc_z = np.ones(n_samples_acc)  # Gravity baseline
    labels = np.zeros(n_samples_acc, dtype=int)

    for i, (start_min, activity) in enumerate(zip(activity_changes, activities)):
        start_idx = int(start_min * 60 * acc_rate)
        if i < len(activity_changes) - 1:
            end_idx = int(activity_changes[i + 1] * 60 * acc_rate)
        else:
            end_idx = n_samples_acc

        # Activity-dependent accelerometer patterns
        if activity == 1:  # Sitting - minimal movement
            acc_x[start_idx:end_idx] = 0.05 * np.random.randn(end_idx - start_idx)
            acc_y[start_idx:end_idx] = 0.05 * np.random.randn(end_idx - start_idx)
            acc_z[start_idx:end_idx] = 1.0 + 0.05 * np.random.randn(end_idx - start_idx)

        elif activity == 2:  # Walking - periodic movement
            t_segment = t_acc[start_idx:end_idx]
            acc_x[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * 2 * t_segment)
            acc_y[start_idx:end_idx] = 0.2 * np.sin(2 * np.pi * 2 * t_segment + np.pi/4)
            acc_z[start_idx:end_idx] = 1.0 + 0.4 * np.sin(2 * np.pi * 2 * t_segment + np.pi/2)
            # Add noise
            acc_x[start_idx:end_idx] += 0.1 * np.random.randn(end_idx - start_idx)
            acc_y[start_idx:end_idx] += 0.1 * np.random.randn(end_idx - start_idx)
            acc_z[start_idx:end_idx] += 0.1 * np.random.randn(end_idx - start_idx)

        elif activity == 3:  # Cycling - high frequency movement
            t_segment = t_acc[start_idx:end_idx]
            acc_x[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 3 * t_segment)
            acc_y[start_idx:end_idx] = 0.4 * np.sin(2 * np.pi * 3 * t_segment + np.pi/3)
            acc_z[start_idx:end_idx] = 1.0 + 0.3 * np.sin(2 * np.pi * 3 * t_segment)
            # Add noise
            acc_x[start_idx:end_idx] += 0.15 * np.random.randn(end_idx - start_idx)
            acc_y[start_idx:end_idx] += 0.15 * np.random.randn(end_idx - start_idx)
            acc_z[start_idx:end_idx] += 0.15 * np.random.randn(end_idx - start_idx)

        labels[start_idx:end_idx] = activity

    # Create data structure matching PPG-DaLiA format
    data = {
        'signal': {
            'wrist': {
                'BVP': ppg_signal,  # Blood Volume Pulse (PPG)
                'ACC': np.column_stack([acc_x, acc_y, acc_z])
            }
        },
        'label': labels,
        'subject': subject_id
    }

    return data


def main():
    """Generate synthetic data for testing."""
    print("Generating synthetic PPG-DaLiA-style data...")
    print("="*60)

    # Create output directory
    output_dir = Path("data/raw/ppg-dalia")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data for 3 subjects
    n_subjects = 3
    duration_minutes = 10  # 10 minutes per subject

    for subject_id in range(1, n_subjects + 1):
        print(f"Generating subject {subject_id}...")

        data = generate_synthetic_subject(subject_id, duration_minutes)

        # Save as pickle file
        output_file = output_dir / f"S{subject_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"  ✓ Saved to {output_file}")
        print(f"    PPG samples: {len(data['signal']['wrist']['BVP'])}")
        print(f"    ACC samples: {len(data['signal']['wrist']['ACC'])}")
        print(f"    Activities: {np.unique(data['label'])}")

    print("\n" + "="*60)
    print(f"✓ Generated {n_subjects} synthetic subjects")
    print(f"✓ Data ready for preprocessing")
    print("\nRun: python3 -m src.ingestion.preprocess")


if __name__ == "__main__":
    main()
