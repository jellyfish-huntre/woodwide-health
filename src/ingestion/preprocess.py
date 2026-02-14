"""
Preprocessing utility for PPG-DaLiA dataset.

Handles loading CSV files, synchronizing multi-rate sensor data (PPG @ 64Hz, ACC @ 32Hz),
and creating rolling window feature vectors for embedding generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import pickle


class PPGDaLiaPreprocessor:
    """
    Preprocessor for PPG-DaLiA dataset.

    The dataset contains:
    - PPG (photoplethysmography): 64 Hz sampling rate
    - Accelerometer (3-axis): 32 Hz sampling rate
    - ECG (reference): 700 Hz sampling rate

    This preprocessor synchronizes the different sampling rates and creates
    feature vectors suitable for multivariate embedding generation.
    """

    # Sensor sampling rates (Hz)
    PPG_RATE = 64
    ACC_RATE = 32
    ECG_RATE = 700

    # Target resampling rate for synchronization
    TARGET_RATE = 32  # Use ACC rate as base (highest common denominator)

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize preprocessor.

        Args:
            data_dir: Directory containing the raw PPG-DaLiA data
        """
        self.data_dir = Path(data_dir)

    def load_subject_data(self, subject_id: int) -> pd.DataFrame:
        """
        Load data for a specific subject.

        Args:
            subject_id: Subject ID (1-15)

        Returns:
            DataFrame with all sensor data

        Raises:
            FileNotFoundError: If subject file doesn't exist
        """
        # PPG-DaLiA files are typically named S1.pkl, S2.pkl, etc.
        subject_file = self.data_dir / "ppg-dalia" / f"S{subject_id}.pkl"

        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")

        with open(subject_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        return data

    def extract_signals(self, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract and structure signals from raw data.

        Handles both real PPG-DaLiA format (BVP as 2D, activity labels in
        'activity' key at 4 Hz) and synthetic format (BVP as 1D, integer
        activity labels in 'label' key at TARGET_RATE).

        Args:
            data: Raw data dictionary from pickle file

        Returns:
            Tuple of (ppg_df, acc_df, labels_df)
        """
        signal = data.get('signal', {})

        # PPG signal — flatten if 2D (real data has shape (n, 1))
        ppg = signal.get('wrist', {}).get('BVP', np.array([]))
        if ppg.ndim > 1:
            ppg = ppg.ravel()
        ppg_df = pd.DataFrame({
            'ppg': ppg,
            'timestamp': np.arange(len(ppg)) / self.PPG_RATE
        })

        # Accelerometer (3-axis)
        acc = signal.get('wrist', {}).get('ACC', np.array([]))
        if acc.ndim == 2:
            acc_df = pd.DataFrame({
                'acc_x': acc[:, 0],
                'acc_y': acc[:, 1],
                'acc_z': acc[:, 2],
                'timestamp': np.arange(len(acc)) / self.ACC_RATE
            })
        else:
            acc_df = pd.DataFrame()

        # Activity labels — real data stores these in 'activity' (at 4 Hz),
        # synthetic data stores integer labels in 'label' (at TARGET_RATE)
        total_duration = len(ppg) / self.PPG_RATE
        if 'activity' in data:
            labels = np.array(data['activity']).ravel().astype(int)
            label_rate = len(labels) / total_duration
        else:
            labels = np.array(data.get('label', np.array([]))).ravel()
            label_rate = self.TARGET_RATE

        labels_df = pd.DataFrame({
            'activity': labels,
            'timestamp': np.arange(len(labels)) / label_rate
        })

        return ppg_df, acc_df, labels_df

    def synchronize_signals(
        self,
        ppg_df: pd.DataFrame,
        acc_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Synchronize PPG and accelerometer signals to a common timeline.

        Different sensors have different sampling rates:
        - PPG: 64 Hz
        - ACC: 32 Hz

        This resamples all signals to TARGET_RATE (32 Hz) using linear interpolation.

        Args:
            ppg_df: PPG data with timestamp
            acc_df: Accelerometer data with timestamp
            labels_df: Activity labels with timestamp

        Returns:
            Synchronized DataFrame with all signals aligned
        """
        # Determine time range (use the shortest signal)
        max_time = min(
            ppg_df['timestamp'].max(),
            acc_df['timestamp'].max(),
            labels_df['timestamp'].max()
        )

        # Create common timeline at TARGET_RATE
        common_timeline = np.arange(0, max_time, 1.0 / self.TARGET_RATE)

        # Resample PPG to common timeline (downsample from 64 Hz to 32 Hz)
        ppg_resampled = np.interp(
            common_timeline,
            ppg_df['timestamp'].values,
            ppg_df['ppg'].values
        )

        # Accelerometer is already at 32 Hz, but align timestamps
        acc_x_resampled = np.interp(
            common_timeline,
            acc_df['timestamp'].values,
            acc_df['acc_x'].values
        )
        acc_y_resampled = np.interp(
            common_timeline,
            acc_df['timestamp'].values,
            acc_df['acc_y'].values
        )
        acc_z_resampled = np.interp(
            common_timeline,
            acc_df['timestamp'].values,
            acc_df['acc_z'].values
        )

        # Resample labels (nearest neighbor for categorical data)
        labels_resampled = np.interp(
            common_timeline,
            labels_df['timestamp'].values,
            labels_df['activity'].values
        ).astype(int)

        synchronized = pd.DataFrame({
            'timestamp': common_timeline,
            'ppg': ppg_resampled,
            'acc_x': acc_x_resampled,
            'acc_y': acc_y_resampled,
            'acc_z': acc_z_resampled,
            'activity': labels_resampled
        })

        return synchronized

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from raw signals.

        Args:
            df: Synchronized dataframe with raw signals

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        # Accelerometer magnitude (overall movement intensity)
        df['acc_magnitude'] = np.sqrt(
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        )

        # Heart rate estimate from PPG (simplified peak detection)
        # In production, use more sophisticated peak detection
        window_size = self.TARGET_RATE * 10  # 10-second window
        df['ppg_rolling_std'] = df['ppg'].rolling(window=window_size, center=True).std()

        return df

    def create_rolling_windows(
        self,
        df: pd.DataFrame,
        window_seconds: float = 30.0,
        stride_seconds: float = 5.0,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create rolling window feature vectors for embedding generation.

        This is the key preprocessing step that transforms time-series into
        the format expected by the Wood Wide API.

        Args:
            df: Synchronized dataframe with all features
            window_seconds: Window size in seconds (default: 30s)
            stride_seconds: Stride between windows in seconds (default: 5s)
            features: List of feature columns to include. If None, uses default set.

        Returns:
            Tuple of (feature_windows, timestamps, labels)
            - feature_windows: shape (n_windows, window_length, n_features)
            - timestamps: shape (n_windows,) - center timestamp of each window
            - labels: shape (n_windows,) - activity label for each window
        """
        if features is None:
            features = ['ppg', 'acc_x', 'acc_y', 'acc_z', 'acc_magnitude']

        window_length = int(window_seconds * self.TARGET_RATE)
        stride_length = int(stride_seconds * self.TARGET_RATE)

        feature_matrix = df[features].values
        timestamps = df['timestamp'].values
        labels = df['activity'].values

        windows = []
        window_timestamps = []
        window_labels = []

        for start_idx in range(0, len(feature_matrix) - window_length + 1, stride_length):
            end_idx = start_idx + window_length

            window = feature_matrix[start_idx:end_idx, :]

            # Center timestamp
            center_idx = start_idx + window_length // 2
            center_time = timestamps[center_idx]

            # Most common label in window (mode)
            window_label = int(np.bincount(labels[start_idx:end_idx].astype(int)).argmax())

            windows.append(window)
            window_timestamps.append(center_time)
            window_labels.append(window_label)

        return (
            np.array(windows),
            np.array(window_timestamps),
            np.array(window_labels)
        )

    def process_subject(
        self,
        subject_id: int,
        window_seconds: float = 30.0,
        stride_seconds: float = 5.0
    ) -> dict:
        """
        Complete preprocessing pipeline for a single subject.

        Args:
            subject_id: Subject ID (1-15)
            window_seconds: Window size for feature vectors
            stride_seconds: Stride between windows

        Returns:
            Dictionary containing processed data:
            {
                'subject_id': int,
                'windows': np.ndarray,
                'timestamps': np.ndarray,
                'labels': np.ndarray,
                'metadata': dict
            }
        """
        print(f"Processing subject {subject_id}...")

        raw_data = self.load_subject_data(subject_id)
        ppg_df, acc_df, labels_df = self.extract_signals(raw_data)
        synchronized = self.synchronize_signals(ppg_df, acc_df, labels_df)
        enriched = self.compute_derived_features(synchronized)
        windows, timestamps, labels = self.create_rolling_windows(
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
                'sampling_rate': self.TARGET_RATE,
                'n_windows': len(windows),
                'window_shape': windows.shape
            }
        }

    def save_processed_data(self, processed_data: dict, output_dir: str = "data/processed"):
        """
        Save processed data to disk.

        Args:
            processed_data: Output from process_subject()
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subject_id = processed_data['subject_id']
        output_file = output_path / f"subject_{subject_id:02d}_processed.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"  ✓ Saved to {output_file}")


def main():
    """Example usage of the preprocessor."""
    preprocessor = PPGDaLiaPreprocessor(data_dir="data/raw")

    # Process first subject as example
    subject_id = 1
    try:
        processed = preprocessor.process_subject(
            subject_id=subject_id,
            window_seconds=30.0,
            stride_seconds=5.0
        )

        preprocessor.save_processed_data(processed)

        # Display info
        print(f"\nProcessed Data Summary:")
        print(f"  Subject: {processed['subject_id']}")
        print(f"  Windows shape: {processed['windows'].shape}")
        print(f"  Timestamps shape: {processed['timestamps'].shape}")
        print(f"  Labels shape: {processed['labels'].shape}")
        print(f"  Unique activities: {np.unique(processed['labels'])}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run download_dataset.py first to download the data.")


if __name__ == "__main__":
    main()
