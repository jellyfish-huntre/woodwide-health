"""
Tests for Wood Wide API client.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.api_client import (
    APIClient,
    MockAPIClient,
    WoodWideAPIError,
    AuthenticationError,
    RateLimitError
)


class TestMockAPIClient:
    """Test suite for MockAPIClient."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock API client."""
        return MockAPIClient(embedding_dim=128)

    @pytest.fixture
    def sample_windows(self):
        """Create sample window data."""
        # 10 windows, 960 timesteps, 5 features
        return np.random.randn(10, 960, 5).astype(np.float32)

    def test_initialization(self, mock_client):
        """Test mock client initializes correctly."""
        assert mock_client.embedding_dim == 128
        assert "mock://" in mock_client.base_url

    def test_health_check(self, mock_client):
        """Test health check endpoint."""
        status = mock_client.check_health()
        assert status["status"] == "healthy"
        assert "version" in status

    def test_get_embedding_info(self, mock_client):
        """Test embedding info endpoint."""
        info = mock_client.get_embedding_info()
        assert "models" in info
        assert len(info["models"]) > 0
        assert info["models"][0]["embedding_dim"] == 128

    def test_generate_embeddings_shape(self, mock_client, sample_windows):
        """Test that embeddings have correct shape."""
        embeddings = mock_client.generate_embeddings(sample_windows, batch_size=5)

        expected_shape = (10, 128)  # 10 windows, 128-dim embeddings
        assert embeddings.shape == expected_shape

    def test_generate_embeddings_normalized(self, mock_client, sample_windows):
        """Test that embeddings are normalized to unit length."""
        embeddings = mock_client.generate_embeddings(sample_windows)

        # Check each embedding is unit length (L2 norm = 1)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(embeddings)), decimal=5)

    def test_generate_single_embedding(self, mock_client, sample_windows):
        """Test single window embedding generation."""
        single_window = sample_windows[0]  # Shape: (960, 5)
        embedding = mock_client.generate_single_embedding(single_window)

        assert embedding.shape == (128,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    def test_batching(self, mock_client, sample_windows):
        """Test that batching doesn't affect results."""
        # Generate with different batch sizes
        embeddings_batch_1 = mock_client.generate_embeddings(sample_windows, batch_size=1)
        embeddings_batch_5 = mock_client.generate_embeddings(sample_windows, batch_size=5)
        embeddings_batch_10 = mock_client.generate_embeddings(sample_windows, batch_size=10)

        # All should produce same results (mock uses fixed seed)
        np.testing.assert_array_almost_equal(embeddings_batch_1, embeddings_batch_5)
        np.testing.assert_array_almost_equal(embeddings_batch_1, embeddings_batch_10)

    def test_context_manager(self, sample_windows):
        """Test using client as context manager."""
        with MockAPIClient(embedding_dim=64) as client:
            embeddings = client.generate_embeddings(sample_windows)
            assert embeddings.shape == (10, 64)

    def test_different_embedding_dims(self, sample_windows):
        """Test different embedding dimensions."""
        for dim in [32, 64, 128, 256]:
            client = MockAPIClient(embedding_dim=dim)
            embeddings = client.generate_embeddings(sample_windows)
            assert embeddings.shape == (10, dim)


class TestAPIClientAuthentication:
    """Test authentication behavior."""

    def test_no_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises AuthenticationError."""
        # Remove API key from environment
        monkeypatch.delenv("WOOD_WIDE_API_KEY", raising=False)

        with pytest.raises(AuthenticationError, match="API key not found"):
            APIClient()

    def test_api_key_from_parameter(self, monkeypatch):
        """Test providing API key as parameter."""
        monkeypatch.delenv("WOOD_WIDE_API_KEY", raising=False)

        # Should not raise if key provided as parameter
        client = APIClient(api_key="test_key_12345")
        assert client.api_key == "test_key_12345"

    def test_api_key_from_environment(self, monkeypatch):
        """Test reading API key from environment."""
        monkeypatch.setenv("WOOD_WIDE_API_KEY", "env_key_67890")

        client = APIClient()
        assert client.api_key == "env_key_67890"


class TestAPIClientConfiguration:
    """Test client configuration options."""

    def test_custom_base_url(self):
        """Test setting custom base URL."""
        client = APIClient(
            api_key="test_key",
            base_url="https://custom.api.com/v2"
        )
        assert client.base_url == "https://custom.api.com/v2"

    def test_custom_timeout(self):
        """Test setting custom timeout."""
        client = APIClient(api_key="test_key", timeout=60)
        assert client.timeout == 60

    def test_custom_retry_settings(self):
        """Test setting custom retry configuration."""
        client = APIClient(
            api_key="test_key",
            max_retries=5,
            retry_delay=2.0
        )
        assert client.max_retries == 5
        assert client.retry_delay == 2.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_client(self):
        return MockAPIClient()

    def test_empty_windows(self, mock_client):
        """Test handling of empty window array."""
        empty_windows = np.array([]).reshape(0, 960, 5)
        embeddings = mock_client.generate_embeddings(empty_windows)
        assert embeddings.shape == (0, 128)

    def test_single_window(self, mock_client):
        """Test processing a single window."""
        single_window = np.random.randn(1, 960, 5).astype(np.float32)
        embeddings = mock_client.generate_embeddings(single_window)
        assert embeddings.shape == (1, 128)

    def test_large_batch(self, mock_client):
        """Test processing a large batch of windows."""
        large_windows = np.random.randn(1000, 960, 5).astype(np.float32)
        embeddings = mock_client.generate_embeddings(large_windows, batch_size=100)
        assert embeddings.shape == (1000, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
