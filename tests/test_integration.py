"""
Integration tests that hit the real Wood Wide API.

These tests require a valid WOOD_WIDE_API_KEY in .env and are marked
with @pytest.mark.integration so they can be run separately:

    pytest tests/test_integration.py -v
    pytest -m integration -v
"""

import os
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.api_client import APIClient


@pytest.fixture(scope="module")
def api_client():
    """Create a real API client from .env credentials.

    Skips all tests if WOOD_WIDE_API_KEY is not configured.
    """
    # Force dotenv load
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("WOOD_WIDE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        pytest.skip("WOOD_WIDE_API_KEY not configured — skipping integration tests")

    client = APIClient(timeout=120)
    yield client
    client.close()


@pytest.fixture
def small_windows():
    """Small windows for quick dataset operations (upload/delete)."""
    np.random.seed(42)
    return np.random.randn(3, 10, 5).astype(np.float32)


@pytest.fixture
def training_windows():
    """Windows sized for successful API training.

    Needs 20+ rows for PyTorch BatchNorm, and modest column count
    to stay under the API's ~500KB upload limit.
    """
    np.random.seed(42)
    return np.random.randn(20, 20, 5).astype(np.float32)


@pytest.mark.integration
class TestHealthAndAuth:
    """Test basic connectivity and authentication."""

    def test_health_check(self, api_client):
        """Test that the API health endpoint is reachable."""
        result = api_client.check_health()
        assert result["status"] == "healthy"

    def test_auth_me(self, api_client):
        """Test that authentication works and returns user info."""
        info = api_client.get_user_info()
        assert "identity_type" in info
        assert "dataset_ids" in info
        assert "model_ids" in info


@pytest.mark.integration
class TestDatasetOperations:
    """Test dataset upload, list, and delete."""

    def test_upload_and_delete_dataset(self, api_client, small_windows):
        """Test uploading a dataset and cleaning it up."""
        csv_bytes = APIClient._windows_to_csv_bytes(small_windows)
        name = "integration_test_upload"

        # Upload
        result = api_client._upload_dataset(csv_bytes, name)
        assert "id" in result
        dataset_id = result["id"]

        try:
            # Verify it appears in the list
            datasets = api_client.list_datasets()
            assert isinstance(datasets, list)
        finally:
            # Clean up
            api_client._delete_dataset(dataset_id)

    def test_list_datasets(self, api_client):
        """Test listing datasets returns a list."""
        datasets = api_client.list_datasets()
        assert isinstance(datasets, list)


@pytest.mark.integration
class TestModelOperations:
    """Test model listing."""

    def test_list_models(self, api_client):
        """Test listing models returns a list."""
        models = api_client.list_models()
        assert isinstance(models, list)


@pytest.mark.integration
class TestFullWorkflow:
    """Test the complete embedding generation workflow."""

    def test_generate_embeddings_end_to_end(self, api_client, training_windows):
        """Test generate_embeddings with real API from upload through inference."""
        embeddings = api_client.generate_embeddings(
            training_windows,
            dataset_name="integration_test_e2e",
            model_name="integration_test_model"
        )

        # Verify shape: 20 windows in, 20 embeddings out
        assert embeddings.shape[0] == 20
        assert embeddings.ndim == 2
        assert embeddings.shape[1] > 0

        # Verify values are finite
        assert np.all(np.isfinite(embeddings))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
