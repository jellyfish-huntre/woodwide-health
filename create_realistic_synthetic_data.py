"""
Create realistic synthetic PPG-DaLiA-like data with proper heart rate patterns.

This version simulates realistic physiological responses:
- Resting HR: 60-80 BPM during sitting
- Exercise HR: 100-140 BPM during cycling/walking
- Gradual transitions between activities
- Activity-dependent accelerometer signals

Usage:
    python create_realistic_synthetic_data.py
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple


def generate_ppg_signal(
    duration_seconds: int,
    sampling_rate: int,
    base_hr: float,
    hr_variability: float = 5.0
) -> np.ndarray:
    """Generate synthetic PPG signal with realistic heart rate.

    Args:
        duration_seconds: Signal duration in seconds
        sampling_rate: Sampling frequency (Hz)
        base_hr: Base heart rate in BPM
        hr_variability: HR variability in BPM

    Returns:
        PPG signal array
    """
    n_samples = duration_seconds * sampling_rate
    t = np.arange(n_samples) / sampling_rate

    # Generate heart rate variation over time
    hr_variation = np.sin(2 * np.pi * 0.05 * t) * hr_variability  # Slow variation
    hr_bpm = base_hr + hr_variation

    # Generate PPG waveform (approximation)
    # PPG has peaks at heartbeats
    ppg = np.zeros(n_samples)
    for i, hr in enumerate(hr_bpm):
        # Heartbeat frequency
        freq = hr / 60.0  # Convert BPM to Hz
        phase = 2 * np.pi * freq * t[i]

        # Simulate PPG waveform (simplified cardiac pulse)
        ppg[i] = np.sin(phase) + 0.3 * np.sin(2 * phase)

    # Add noise
    noise = np.random.randn(n_samples) * 0.1
    ppg = ppg + noise

    # Normalize to reasonable range
    ppg = (ppg - ppg.mean()) / ppg.std()

    return ppg.astype(np.float32)


def generate_accelerometer_signal(
    duration_seconds: int,
    sampling_rate: int,
    activity_intensity: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic accelerometer signals (X, Y, Z).

    Args:
        duration_seconds: Signal duration in seconds
        sampling_rate: Sampling frequency (Hz)
        activity_intensity: Activity intensity (0=rest, 1=moderate, 2=vigorous)

    Returns:
        Tuple of (acc_x, acc_y, acc_z)
    """
    n_samples = duration_seconds * sampling_rate
    t = np.arange(n_samples) / sampling_rate

    # Activity-dependent frequency and amplitude
    if activity_intensity == 0:  # Sitting/resting
        freq = 0.1  # Very low frequency movement
        amplitude = 0.05
    elif activity_intensity == 1:  # Moderate (walking)
        freq = 2.0  # Step frequency ~2 Hz
        amplitude = 1.0
    else:  # Vigorous (cycling, stairs)
        freq = 2.5
        amplitude = 1.5

    # Generate acceleration components
    acc_x = amplitude * np.sin(2 * np.pi * freq * t + np.random.randn() * 0.1)
    acc_y = amplitude * np.sin(2 * np.pi * freq * t + np.pi/3 + np.random.randn() * 0.1)
    acc_z = amplitude * np.sin(2 * np.pi * freq * t + 2*np.pi/3 + np.random.randn() * 0.1)

    # Add gravity component to z-axis
    acc_z = acc_z + 9.81

    # Add noise
    acc_x = acc_x + np.random.randn(n_samples) * 0.1
    acc_y = acc_y + np.random.randn(n_samples) * 0.1
    acc_z = acc_z + np.random.randn(n_samples) * 0.1

    return acc_x.astype(np.float32), acc_y.astype(np.float32), acc_z.astype(np.float32)


def create_activity_sequence() -> list:
    """Create a realistic sequence of activities.

    Returns:
        List of (activity_id, duration_seconds, hr_bpm, intensity) tuples
    """
    # Activity ID mapping:
    # 1: Sitting, 2: Cycling, 3: Walking, 4: Ascending stairs,
    # 5: Descending stairs, 6: Vacuum cleaning, 7: Ironing

    sequence = [
        (1, 120, 70, 0),   # Sitting - 2 min - resting HR
        (2, 180, 130, 2),  # Cycling - 3 min - high HR
        (1, 60, 75, 0),    # Sitting - 1 min - recovery
        (3, 120, 110, 1),  # Walking - 2 min - moderate HR
        (1, 60, 72, 0),    # Sitting - 1 min - recovery
        (4, 60, 135, 2),   # Ascending stairs - 1 min - very high HR
        (1, 60, 80, 0),    # Sitting - 1 min - recovery
    ]

    return sequence


def create_subject_data(
    subject_id: int,
    ppg_rate: int = 64,
    acc_rate: int = 32
) -> Dict:
    """Create realistic synthetic data for one subject.

    Args:
        subject_id: Subject identifier
        ppg_rate: PPG sampling rate (Hz)
        acc_rate: Accelerometer sampling rate (Hz)

    Returns:
        Dictionary with subject data
    """
    activity_sequence = create_activity_sequence()
    total_duration = sum(duration for _, duration, _, _ in activity_sequence)

    print(f"  Generating subject {subject_id}:")
    print(f"    Total duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")

    # Initialize arrays
    ppg_total = []
    acc_x_total = []
    acc_y_total = []
    acc_z_total = []
    timestamps_ppg = []
    timestamps_acc = []
    labels = []

    current_time = 0.0

    for activity_id, duration, hr_bpm, intensity in activity_sequence:
        print(f"    {duration}s - Activity {activity_id} - HR {hr_bpm} BPM")

        # Generate PPG
        ppg_segment = generate_ppg_signal(duration, ppg_rate, hr_bpm)
        ppg_total.append(ppg_segment)

        # Generate timestamps for PPG
        ppg_times = current_time + np.arange(len(ppg_segment)) / ppg_rate
        timestamps_ppg.append(ppg_times)

        # Generate accelerometer
        acc_x, acc_y, acc_z = generate_accelerometer_signal(duration, acc_rate, intensity)
        acc_x_total.append(acc_x)
        acc_y_total.append(acc_y)
        acc_z_total.append(acc_z)

        # Generate timestamps for accelerometer
        acc_times = current_time + np.arange(len(acc_x)) / acc_rate
        timestamps_acc.append(acc_times)

        # Create labels (one per accelerometer sample)
        labels.append(np.full(len(acc_x), activity_id))

        current_time += duration

    # Concatenate all segments
    ppg = np.concatenate(ppg_total)
    acc_x = np.concatenate(acc_x_total)
    acc_y = np.concatenate(acc_y_total)
    acc_z = np.concatenate(acc_z_total)
    timestamps_ppg = np.concatenate(timestamps_ppg)
    timestamps_acc = np.concatenate(timestamps_acc)
    labels = np.concatenate(labels)

    return {
        'subject_id': subject_id,
        'ppg': ppg,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'timestamps_ppg': timestamps_ppg,
        'timestamps_acc': timestamps_acc,
        'labels': labels,
        'ppg_sampling_rate': ppg_rate,
        'acc_sampling_rate': acc_rate
    }


def main():
    """Generate realistic synthetic data for multiple subjects."""
    print("Generating realistic synthetic PPG-DaLiA data...")
    print("=" * 80)

    output_dir = Path("data/PPGDaLiA")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 3

    for subject_id in range(1, n_subjects + 1):
        data = create_subject_data(subject_id)

        # Save as pickle
        output_file = output_dir / f"S{subject_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"  ✓ Saved to {output_file}")
        print(f"    PPG samples: {len(data['ppg'])} ({data['ppg_sampling_rate']} Hz)")
        print(f"    ACC samples: {len(data['acc_x'])} ({data['acc_sampling_rate']} Hz)")
        print()

    print("=" * 80)
    print(f"✓ Generated {n_subjects} subjects with realistic HR patterns")
    print()
    print("Key features:")
    print("  • Resting HR: 70-80 BPM during sitting")
    print("  • Exercise HR: 110-135 BPM during activity")
    print("  • Realistic activity sequences with recovery periods")
    print("  • Activity-dependent accelerometer patterns")
    print()
    print("Next steps:")
    print("  1. Run preprocessing: python -m src.ingestion.preprocess")
    print("  2. Run baseline detection: python baseline_threshold_detection.py 1 --threshold 100")
    print()


if __name__ == "__main__":
    main()
