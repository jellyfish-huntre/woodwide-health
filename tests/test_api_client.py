"""
Tests for Wood Wide API client.
"""

import numpy as np
import pytest
import requests
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

from src.embeddings.api_client import (
    APIClient,
    MockAPIClient,
    WoodWideAPIError,
    AuthenticationError,
    RateLimitError,
    SSLConnectionError,
    TrainingTimeoutError,
    _SSLAdapter
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

    def test_get_user_info(self, mock_client):
        """Test user info endpoint."""
        info = mock_client.get_user_info()
        assert "identity_type" in info
        assert "dataset_ids" in info
        assert "model_ids" in info

    def test_generate_embeddings_shape(self, mock_client, sample_windows):
        """Test that embeddings have correct shape."""
        embeddings = mock_client.generate_embeddings(sample_windows)

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

    def test_deterministic_results(self, mock_client, sample_windows):
        """Test that same input produces same embeddings."""
        embeddings_1 = mock_client.generate_embeddings(sample_windows)
        embeddings_2 = mock_client.generate_embeddings(sample_windows)

        np.testing.assert_array_almost_equal(embeddings_1, embeddings_2)

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

    def test_default_base_url(self, monkeypatch):
        """Test default base URL points to beta."""
        monkeypatch.delenv("WOOD_WIDE_API_URL", raising=False)
        client = APIClient(api_key="test_key")
        assert client.base_url == "https://beta.woodwide.ai"

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

    def test_training_timeout_setting(self):
        """Test setting custom training timeout."""
        client = APIClient(api_key="test_key", training_timeout=300)
        assert client.training_timeout == 300

    def test_poll_interval_setting(self):
        """Test setting custom poll interval."""
        client = APIClient(api_key="test_key", poll_interval=10.0)
        assert client.poll_interval == 10.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_client(self):
        return MockAPIClient()

    def test_empty_windows(self, mock_client):
        """Test handling of empty window array."""
        empty_windows = np.array([]).reshape(0, 960, 5)
        embeddings = mock_client.generate_embeddings(empty_windows)
        assert embeddings.shape[0] == 0
        assert embeddings.ndim == 2

    def test_single_window(self, mock_client):
        """Test processing a single window."""
        single_window = np.random.randn(1, 960, 5).astype(np.float32)
        embeddings = mock_client.generate_embeddings(single_window)
        assert embeddings.shape == (1, 128)

    def test_large_batch(self, mock_client):
        """Test processing a large batch of windows."""
        large_windows = np.random.randn(100, 960, 5).astype(np.float32)
        embeddings = mock_client.generate_embeddings(large_windows)
        assert embeddings.shape == (100, 128)


class TestSSLConfiguration:
    """Test SSL/TLS configuration."""

    def test_ssl_adapter_mounted(self):
        """Test that HTTPS adapter with custom SSL context is mounted."""
        client = APIClient(api_key="test_key")
        adapter = client.session.get_adapter("https://api.woodwide.ai")
        assert isinstance(adapter, _SSLAdapter)

    def test_ssl_context_minimum_tls_version(self):
        """Test that SSL context enforces TLS 1.2 minimum."""
        import ssl
        ctx = APIClient._create_ssl_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_custom_ciphers_from_parameter(self):
        """Test providing custom ciphers via parameter."""
        client = APIClient(api_key="test_key", ssl_ciphers="DEFAULT:@SECLEVEL=0")
        assert client.ssl_ciphers == "DEFAULT:@SECLEVEL=0"

    def test_custom_ciphers_from_environment(self, monkeypatch):
        """Test reading ciphers from environment variable."""
        monkeypatch.setenv("WOOD_WIDE_SSL_CIPHERS", "DEFAULT:@SECLEVEL=0")
        client = APIClient(api_key="test_key")
        assert client.ssl_ciphers == "DEFAULT:@SECLEVEL=0"

    def test_ssl_handshake_failure_raises_ssl_connection_error(self):
        """Test that SSL handshake failures raise SSLConnectionError immediately."""
        client = APIClient(api_key="test_key")
        with patch.object(
            client.session, "request",
            side_effect=requests.exceptions.SSLError("SSLV3_ALERT_HANDSHAKE_FAILURE")
        ):
            with pytest.raises(SSLConnectionError, match="SSL handshake failed"):
                client._make_request("GET", "/health")

    def test_ssl_handshake_failure_not_retried(self):
        """Test that handshake failures fail immediately without retrying."""
        client = APIClient(api_key="test_key", max_retries=3)
        call_count = 0

        def counting_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.SSLError("SSLV3_ALERT_HANDSHAKE_FAILURE")

        with patch.object(client.session, "request", side_effect=counting_side_effect):
            with pytest.raises(SSLConnectionError):
                client._make_request("GET", "/health")

        assert call_count == 1


class TestWindowsToCSV:
    """Test CSV conversion of window data."""

    def test_csv_shape(self):
        """Test that CSV has correct number of rows and columns."""
        windows = np.random.randn(5, 960, 5).astype(np.float32)
        csv_bytes = APIClient._windows_to_csv_bytes(windows)
        lines = csv_bytes.decode("utf-8").strip().split("\n")
        assert len(lines) == 6  # 1 header + 5 data rows
        assert len(lines[0].split(",")) == 4800  # 960 * 5 columns

    def test_csv_header(self):
        """Test CSV header format."""
        windows = np.random.randn(2, 10, 3).astype(np.float32)
        csv_bytes = APIClient._windows_to_csv_bytes(windows)
        header = csv_bytes.decode("utf-8").split("\n")[0].strip()
        cols = header.split(",")
        assert cols[0] == "f_0"
        assert cols[-1] == "f_29"  # 10 * 3 - 1
        assert len(cols) == 30

    def test_csv_roundtrip(self):
        """Test that values survive CSV conversion."""
        windows = np.array([[[1.5, 2.5], [3.5, 4.5]]]).astype(np.float32)
        csv_bytes = APIClient._windows_to_csv_bytes(windows)
        data_line = csv_bytes.decode("utf-8").strip().split("\n")[1]
        values = [float(v) for v in data_line.split(",")]
        np.testing.assert_array_almost_equal(values, [1.5, 2.5, 3.5, 4.5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
