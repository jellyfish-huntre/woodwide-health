"""
Preprocess multiple subjects from the PPG-DaLiA dataset.

Usage:
    python preprocess_subjects.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.ingestion.preprocess import PPGDaLiaPreprocessor


def process_synthetic_subject(subject_id: int, window_seconds: float = 30.0, stride_seconds: float = 5.0):
    """
    Process synthetic data subjects (which have a different format than real PPG-DaLiA).

    Args:
        subject_id: Subject ID
        window_seconds: Window length in seconds
        stride_seconds: Stride between windows in seconds

    Returns:
        Processed data dictionary
    """
    print(f"Processing subject {subject_id}...")

    # Load synthetic data
    data_file = Path(f"data/PPGDaLiA/S{subject_id}.pkl")
    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)

    # Create preprocessor
    preprocessor = PPGDaLiaPreprocessor()

    # Convert to DataFrames (synthetic data already has extracted signals)
    ppg_df = pd.DataFrame({
        'ppg': raw_data['ppg'],
        'timestamp': raw_data['timestamps_ppg']
    })

    acc_df = pd.DataFrame({
        'acc_x': raw_data['acc_x'],
        'acc_y': raw_data['acc_y'],
        'acc_z': raw_data['acc_z'],
        'timestamp': raw_data['timestamps_acc']
    })

    labels_df = pd.DataFrame({
        'activity': raw_data['labels'],
        'timestamp': raw_data['timestamps_acc']
    })

    # Synchronize signals
    synchronized = preprocessor.synchronize_signals(ppg_df, acc_df, labels_df)

    # Compute derived features
    enriched = preprocessor.compute_derived_features(synchronized)

    # Create windows
    windows, timestamps, labels = preprocessor.create_rolling_windows(
        enriched,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds
    )

    print(f"  ✓ Created {len(windows)} windows of {window_seconds}s each")

    return {
        'subject_id': subject_id,
        'windows': windows,
        'timestamps': timestamps,
        'labels': labels,
        'metadata': {
            'window_seconds': window_seconds,
            'stride_seconds': stride_seconds,
            'sampling_rate': preprocessor.TARGET_RATE,
            'n_windows': len(windows),
            'window_shape': windows.shape
        }
    }


def save_processed_data(processed_data: dict, output_dir: str = "data/processed"):
    """Save processed data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subject_id = processed_data['subject_id']
    output_file = output_path / f"subject_{subject_id:02d}_processed.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"  ✓ Saved to {output_file}")


def main():
    """Preprocess subjects 2 and 3."""
    # Subjects to process
    subjects = [2, 3]

    for subject_id in subjects:
        try:
            processed = process_synthetic_subject(
                subject_id=subject_id,
                window_seconds=30.0,
                stride_seconds=5.0
            )

            # Save
            save_processed_data(processed)

            # Display info
            print(f"  ✓ Subject {subject_id} processed successfully")
            print(f"    Windows: {processed['windows'].shape[0]}")
            print(f"    Unique activities: {np.unique(processed['labels'])}")

        except FileNotFoundError as e:
            print(f"  ✗ Error processing subject {subject_id}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error for subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Preprocessing complete!")

    # Show all processed subjects
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        processed_files = sorted(processed_dir.glob("subject_*_processed.pkl"))
        print(f"\nTotal processed subjects: {len(processed_files)}")
        for f in processed_files:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
