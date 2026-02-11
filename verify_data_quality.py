"""
Data quality verification script.

Loads processed data and verifies:
- No null/NaN values exist
- Data ranges are reasonable
- All windows have correct shape
- Statistics summary
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys


def load_processed_data(subject_id: int, data_dir: str = "data/processed"):
    """Load processed subject data."""
    file_path = Path(data_dir) / f"subject_{subject_id:02d}_processed.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def verify_no_nulls(windows: np.ndarray, timestamps: np.ndarray, labels: np.ndarray):
    """
    Verify no null/NaN values exist in the data.

    Returns:
        bool: True if no nulls found, False otherwise
    """
    print("\n" + "="*70)
    print("NULL VALUE CHECK")
    print("="*70)

    windows_nulls = np.isnan(windows).sum()
    timestamps_nulls = np.isnan(timestamps).sum()
    labels_nulls = np.isnan(labels).sum()

    print(f"Windows: {windows_nulls} null values")
    print(f"Timestamps: {timestamps_nulls} null values")
    print(f"Labels: {labels_nulls} null values")

    total_nulls = windows_nulls + timestamps_nulls + labels_nulls

    if total_nulls == 0:
        print("\n✓ PASSED: No null values found in dataset")
        return True
    else:
        print(f"\n✗ FAILED: Found {total_nulls} null values")
        return False


def verify_data_ranges(windows: np.ndarray):
    """
    Verify data values are within reasonable ranges.

    Returns:
        bool: True if ranges are reasonable
    """
    print("\n" + "="*70)
    print("DATA RANGE CHECK")
    print("="*70)

    feature_names = ['PPG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_MAG']

    all_reasonable = True

    for i, feature in enumerate(feature_names):
        feature_data = windows[:, :, i].flatten()

        min_val = feature_data.min()
        max_val = feature_data.max()
        mean_val = feature_data.mean()
        std_val = feature_data.std()

        print(f"\n{feature}:")
        print(f"  Min:    {min_val:>10.4f}")
        print(f"  Max:    {max_val:>10.4f}")
        print(f"  Mean:   {mean_val:>10.4f}")
        print(f"  Std:    {std_val:>10.4f}")

        # Check for reasonable ranges (these are heuristics)
        if feature == 'PPG':
            # PPG values should be normalized, roughly -5 to 5 range
            if abs(min_val) > 100 or abs(max_val) > 100:
                print(f"  ⚠ WARNING: {feature} values seem extreme")
                all_reasonable = False
        elif feature.startswith('ACC'):
            # Accelerometer should be roughly -10 to 10 g
            if abs(min_val) > 20 or abs(max_val) > 20:
                print(f"  ⚠ WARNING: {feature} values seem extreme")
                all_reasonable = False

    if all_reasonable:
        print("\n✓ PASSED: All data ranges are reasonable")
    else:
        print("\n⚠ WARNING: Some data ranges may be unusual")

    return all_reasonable


def verify_window_shapes(windows: np.ndarray, expected_shape: tuple):
    """
    Verify all windows have correct shape.

    Returns:
        bool: True if all shapes correct
    """
    print("\n" + "="*70)
    print("WINDOW SHAPE CHECK")
    print("="*70)

    actual_shape = windows.shape
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape:   {actual_shape}")

    if actual_shape == expected_shape:
        print("\n✓ PASSED: Window shapes match expected")
        return True
    else:
        print(f"\n✗ FAILED: Shape mismatch")
        return False


def generate_dataframe_summary(windows: np.ndarray, timestamps: np.ndarray, labels: np.ndarray):
    """
    Generate a summary DataFrame for easier inspection.

    Returns:
        DataFrame with aggregated window statistics
    """
    print("\n" + "="*70)
    print("DATAFRAME SUMMARY")
    print("="*70)

    feature_names = ['PPG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_MAG']

    # Create summary statistics for each window
    summary_data = []

    for window_idx in range(len(windows)):
        window = windows[window_idx]
        timestamp = timestamps[window_idx]
        label = labels[window_idx]

        window_summary = {
            'window_idx': window_idx,
            'timestamp': timestamp,
            'activity': label
        }

        # Add mean and std for each feature
        for feat_idx, feat_name in enumerate(feature_names):
            feature_data = window[:, feat_idx]
            window_summary[f'{feat_name}_mean'] = feature_data.mean()
            window_summary[f'{feat_name}_std'] = feature_data.std()

        summary_data.append(window_summary)

    df = pd.DataFrame(summary_data)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nDescriptive statistics:")
    print(df.describe())

    # Check for nulls in DataFrame
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print(f"\n✓ No null values in DataFrame")
    else:
        print(f"\n✗ Null values found:")
        print(null_counts[null_counts > 0])

    return df


def main():
    """Run data quality verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify processed data quality")
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID to verify (e.g., 1)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed data"
    )

    args = parser.parse_args()

    print("="*70)
    print(f"DATA QUALITY VERIFICATION - Subject {args.subject_id}")
    print("="*70)

    try:
        # Load data
        print(f"\nLoading data from: {args.data_dir}")
        data = load_processed_data(args.subject_id, args.data_dir)

        windows = data['windows']
        timestamps = data['timestamps']
        labels = data['labels']
        metadata = data['metadata']

        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Run verification checks
        results = []

        # 1. Check for nulls
        results.append(verify_no_nulls(windows, timestamps, labels))

        # 2. Check data ranges
        results.append(verify_data_ranges(windows))

        # 3. Check window shapes
        expected_shape = (
            metadata['n_windows'],
            int(metadata['window_seconds'] * metadata['sampling_rate']),
            5  # Number of features
        )
        results.append(verify_window_shapes(windows, expected_shape))

        # 4. Generate DataFrame summary
        df = generate_dataframe_summary(windows, timestamps, labels)

        # Final summary
        print("\n" + "="*70)
        print("FINAL VERIFICATION SUMMARY")
        print("="*70)

        checks_passed = sum(results)
        total_checks = len(results)

        print(f"\nChecks passed: {checks_passed}/{total_checks}")

        if all(results):
            print("\n✓ ALL CHECKS PASSED - Data quality verified!")
            return 0
        else:
            print("\n⚠ SOME CHECKS FAILED - Review warnings above")
            return 1

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please run preprocessing first:")
        print("  python3 -m src.ingestion.preprocess")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
