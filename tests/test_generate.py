"""
Tests for embedding generation functions.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.generate import (
    send_windows_to_woodwide,
    _validate_windows,
    _validate_embeddings,
    process_subject_windows,
    load_embeddings
)


class TestSendWindowsToWoodwide:
    """Test the main send_windows_to_woodwide function."""

    @pytest.fixture
    def valid_windows(self):
        """Create valid window data."""
        # 10 windows, 960 timesteps, 5 features
        return np.random.randn(10, 960, 5).astype(np.float32)

    def test_basic_usage_mock(self, valid_windows):
        """Test basic usage with mock API."""
        embeddings, metadata = send_windows_to_woodwide(
            valid_windows,
            batch_size=5,
            embedding_dim=128,
            use_mock=True
        )

        # Check embeddings shape
        assert embeddings.shape == (10, 128)

        # Check metadata
        assert metadata['n_windows'] == 10
        assert metadata['embedding_dim'] == 128
        assert metadata['batch_size'] == 5
        assert metadata['mock'] is True
        assert 'processing_time_seconds' in metadata
        assert 'windows_per_second' in metadata

    def test_different_embedding_dims(self, valid_windows):
        """Test with different embedding dimensions."""
        for dim in [32, 64, 128, 256]:
            embeddings, _ = send_windows_to_woodwide(
                valid_windows,
                embedding_dim=dim,
                use_mock=True
            )
            assert embeddings.shape == (10, dim)

    def test_different_batch_sizes(self, valid_windows):
        """Test with different batch sizes."""
        for batch_size in [1, 5, 10, 20]:
            embeddings, metadata = send_windows_to_woodwide(
                valid_windows,
                batch_size=batch_size,
                use_mock=True
            )
            assert embeddings.shape[0] == 10
            assert metadata['batch_size'] == batch_size

    def test_single_window(self):
        """Test with a single window."""
        single_window = np.random.randn(1, 960, 5).astype(np.float32)
        embeddings, metadata = send_windows_to_woodwide(
            single_window,
            use_mock=True
        )
        assert embeddings.shape[0] == 1
        assert metadata['n_windows'] == 1

    def test_large_batch(self):
        """Test with a large number of windows."""
        large_windows = np.random.randn(100, 960, 5).astype(np.float32)
        embeddings, metadata = send_windows_to_woodwide(
            large_windows,
            batch_size=32,
            use_mock=True
        )
        assert embeddings.shape == (100, 128)
        assert metadata['n_windows'] == 100


class TestValidateWindows:
    """Test input validation."""

    def test_valid_windows_pass(self):
        """Test that valid windows pass validation."""
        valid_windows = np.random.randn(10, 960, 5)
        _validate_windows(valid_windows)  # Should not raise

    def test_wrong_dimensions_fail(self):
        """Test that wrong dimensions fail."""
        # 2D array
        with pytest.raises(ValueError, match="must be 3D array"):
            _validate_windows(np.random.randn(10, 960))

        # 4D array
        with pytest.raises(ValueError, match="must be 3D array"):
            _validate_windows(np.random.randn(10, 960, 5, 1))

    def test_zero_windows_fail(self):
        """Test that zero windows fail."""
        empty_windows = np.array([]).reshape(0, 960, 5)
        with pytest.raises(ValueError, match="No windows to process"):
            _validate_windows(empty_windows)

    def test_nan_values_fail(self):
        """Test that NaN values fail."""
        windows_with_nan = np.random.randn(10, 960, 5)
        windows_with_nan[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="null/NaN values"):
            _validate_windows(windows_with_nan)

    def test_inf_values_fail(self):
        """Test that infinite values fail."""
        windows_with_inf = np.random.randn(10, 960, 5)
        windows_with_inf[0, 0, 0] = np.inf

        with pytest.raises(ValueError, match="infinite values"):
            _validate_windows(windows_with_inf)

    def test_wrong_features_warning(self, caplog):
        """Test that wrong number of features generates warning."""
        # 3 features instead of 5
        windows_3_features = np.random.randn(10, 960, 3)
        _validate_windows(windows_3_features)

        # Should have logged a warning
        assert "Expected 5 features" in caplog.text

    def test_extreme_values_warning(self, caplog):
        """Test that extreme values generate warning."""
        windows_extreme = np.random.randn(10, 960, 5)
        windows_extreme[0, 0, 0] = 10000  # Extreme value

        _validate_windows(windows_extreme)

        # Should have logged a warning
        assert "extreme values" in caplog.text.lower()


class TestValidateEmbeddings:
    """Test embedding validation."""

    def test_valid_embeddings_pass(self):
        """Test that valid embeddings pass."""
        # Unit-normalized embeddings
        embeddings = np.random.randn(10, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        _validate_embeddings(embeddings, expected_n_windows=10)

    def test_wrong_dimensions_fail(self):
        """Test that wrong dimensions fail."""
        # 1D array
        with pytest.raises(ValueError, match="must be 2D array"):
            _validate_embeddings(np.random.randn(10), expected_n_windows=10)

        # 3D array
        with pytest.raises(ValueError, match="must be 2D array"):
            _validate_embeddings(np.random.randn(10, 128, 1), expected_n_windows=10)

    def test_wrong_count_fail(self):
        """Test that wrong number of embeddings fails."""
        embeddings = np.random.randn(10, 128)

        with pytest.raises(ValueError, match="Expected 5 embeddings, got 10"):
            _validate_embeddings(embeddings, expected_n_windows=5)

    def test_nan_values_fail(self):
        """Test that NaN values fail."""
        embeddings = np.random.randn(10, 128)
        embeddings[0, 0] = np.nan

        with pytest.raises(ValueError, match="null/NaN values"):
            _validate_embeddings(embeddings, expected_n_windows=10)

    def test_non_normalized_warning(self, caplog):
        """Test that non-normalized embeddings generate warning."""
        # Not normalized
        embeddings = np.random.randn(10, 128) * 10

        _validate_embeddings(embeddings, expected_n_windows=10)

        # Should have logged a warning
        assert "not be unit-normalized" in caplog.text


class TestProcessSubjectWindows:
    """Test processing complete subject workflow."""

    def test_process_subject_1_mock(self):
        """Test processing subject 1 with mock API."""
        # This test requires preprocessed data to exist
        embeddings, metadata = process_subject_windows(
            subject_id=1,
            data_dir="data/processed",
            output_dir="data/embeddings_test",
            batch_size=32,
            use_mock=True
        )

        # Check embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2

        # Check metadata
        assert metadata['subject_id'] == 1
        assert 'embeddings_shape' in metadata
        assert 'timestamps' in metadata
        assert 'labels' in metadata
        assert 'generation_metadata' in metadata

        # Check files were created
        output_dir = Path("data/embeddings_test")
        assert (output_dir / "subject_01_embeddings.npy").exists()
        assert (output_dir / "subject_01_metadata.pkl").exists()

    def test_missing_data_raises_error(self):
        """Test that missing data raises appropriate error."""
        with pytest.raises(FileNotFoundError, match="Preprocessed data not found"):
            process_subject_windows(
                subject_id=999,  # Non-existent subject
                use_mock=True
            )


class TestLoadEmbeddings:
    """Test loading embeddings."""

    def test_load_after_save(self):
        """Test loading embeddings after saving."""
        # First generate and save
        process_subject_windows(
            subject_id=1,
            output_dir="data/embeddings_test",
            use_mock=True
        )

        # Then load
        embeddings, metadata = load_embeddings(
            subject_id=1,
            data_dir="data/embeddings_test"
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert metadata is not None
        assert metadata['subject_id'] == 1

    def test_load_nonexistent_raises_error(self):
        """Test that loading non-existent embeddings raises error."""
        with pytest.raises(FileNotFoundError, match="Embeddings not found"):
            load_embeddings(subject_id=999)


class TestValidationSkipping:
    """Test validation can be skipped."""

    def test_skip_validation(self):
        """Test that validation can be skipped."""
        # Create invalid windows (NaN values)
        invalid_windows = np.random.randn(10, 960, 5)
        invalid_windows[0, 0, 0] = np.nan

        # With validation, should fail
        with pytest.raises(ValueError):
            send_windows_to_woodwide(
                invalid_windows,
                validate_input=True,
                use_mock=True
            )

        # Without validation, should proceed (mock will handle it)
        embeddings, _ = send_windows_to_woodwide(
            invalid_windows,
            validate_input=False,
            use_mock=True
        )
        assert embeddings is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
