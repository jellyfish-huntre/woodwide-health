"""
Tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.preprocess import PPGDaLiaPreprocessor


class TestPPGDaLiaPreprocessor:
    """Test suite for PPG-DaLiA preprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return PPGDaLiaPreprocessor(data_dir="data/raw")

    @pytest.fixture
    def mock_ppg_data(self):
        """Create mock PPG data."""
        n_samples = 64 * 60  # 1 minute at 64 Hz
        return pd.DataFrame({
            'ppg': np.sin(np.linspace(0, 10 * np.pi, n_samples)),
            'timestamp': np.arange(n_samples) / 64.0
        })

    @pytest.fixture
    def mock_acc_data(self):
        """Create mock accelerometer data."""
        n_samples = 32 * 60  # 1 minute at 32 Hz
        return pd.DataFrame({
            'acc_x': np.random.randn(n_samples) * 0.1,
            'acc_y': np.random.randn(n_samples) * 0.1,
            'acc_z': np.random.randn(n_samples) * 0.1 + 1.0,  # Gravity
            'timestamp': np.arange(n_samples) / 32.0
        })

    @pytest.fixture
    def mock_labels(self):
        """Create mock activity labels."""
        n_samples = 32 * 60  # 1 minute at 32 Hz
        # Simulate activity changes
        labels = np.ones(n_samples, dtype=int)
        labels[n_samples//2:] = 2
        return pd.DataFrame({
            'activity': labels,
            'timestamp': np.arange(n_samples) / 32.0
        })

    def test_synchronize_signals(self, preprocessor, mock_ppg_data, mock_acc_data, mock_labels):
        """Test signal synchronization."""
        synchronized = preprocessor.synchronize_signals(
            mock_ppg_data,
            mock_acc_data,
            mock_labels
        )

        # Check all required columns exist
        assert 'timestamp' in synchronized.columns
        assert 'ppg' in synchronized.columns
        assert 'acc_x' in synchronized.columns
        assert 'acc_y' in synchronized.columns
        assert 'acc_z' in synchronized.columns
        assert 'activity' in synchronized.columns

        # Check sampling rate is TARGET_RATE (32 Hz)
        time_diff = synchronized['timestamp'].diff().dropna()
        expected_interval = 1.0 / preprocessor.TARGET_RATE
        np.testing.assert_almost_equal(time_diff.mean(), expected_interval, decimal=4)

    def test_compute_derived_features(self, preprocessor, mock_ppg_data, mock_acc_data, mock_labels):
        """Test derived feature computation."""
        synchronized = preprocessor.synchronize_signals(
            mock_ppg_data,
            mock_acc_data,
            mock_labels
        )
        enriched = preprocessor.compute_derived_features(synchronized)

        # Check derived features exist
        assert 'acc_magnitude' in enriched.columns
        assert 'ppg_rolling_std' in enriched.columns

        # Check accelerometer magnitude is computed correctly
        expected_mag = np.sqrt(
            enriched['acc_x']**2 +
            enriched['acc_y']**2 +
            enriched['acc_z']**2
        )
        np.testing.assert_array_almost_equal(
            enriched['acc_magnitude'].values,
            expected_mag.values,
            decimal=5
        )

    def test_create_rolling_windows(self, preprocessor, mock_ppg_data, mock_acc_data, mock_labels):
        """Test rolling window creation."""
        synchronized = preprocessor.synchronize_signals(
            mock_ppg_data,
            mock_acc_data,
            mock_labels
        )
        enriched = preprocessor.compute_derived_features(synchronized)

        window_seconds = 10.0
        stride_seconds = 5.0

        windows, timestamps, labels = preprocessor.create_rolling_windows(
            enriched,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds
        )

        # Check shapes
        window_length = int(window_seconds * preprocessor.TARGET_RATE)
        n_features = 5  # ppg, acc_x, acc_y, acc_z, acc_magnitude

        assert windows.shape[1] == window_length
        assert windows.shape[2] == n_features
        assert len(timestamps) == len(windows)
        assert len(labels) == len(windows)

        # Check windows are non-overlapping correctly
        stride_samples = int(stride_seconds * preprocessor.TARGET_RATE)
        expected_n_windows = (len(enriched) - window_length) // stride_samples + 1
        assert len(windows) == expected_n_windows or len(windows) == expected_n_windows + 1

    def test_window_labels(self, preprocessor, mock_ppg_data, mock_acc_data, mock_labels):
        """Test that window labels are correctly assigned."""
        synchronized = preprocessor.synchronize_signals(
            mock_ppg_data,
            mock_acc_data,
            mock_labels
        )
        enriched = preprocessor.compute_derived_features(synchronized)

        windows, timestamps, labels = preprocessor.create_rolling_windows(
            enriched,
            window_seconds=5.0,
            stride_seconds=5.0
        )

        # First half should be mostly label 1, second half label 2
        first_half_labels = labels[:len(labels)//2]
        second_half_labels = labels[len(labels)//2:]

        assert np.mean(first_half_labels == 1) > 0.8
        assert np.mean(second_half_labels == 2) > 0.8


def test_preprocessor_initialization():
    """Test preprocessor can be initialized."""
    preprocessor = PPGDaLiaPreprocessor(data_dir="test_dir")
    assert preprocessor.data_dir == Path("test_dir")
    assert preprocessor.TARGET_RATE == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
