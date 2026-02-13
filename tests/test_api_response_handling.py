"""
Tests for API response handling with mocked HTTP requests.

These tests verify that the APIClient correctly handles the multi-step
workflow (upload → train → poll → infer) against the Wood Wide API.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.api_client import (
    APIClient,
    WoodWideAPIError,
    TrainingTimeoutError
)


class TestWorkflowOrchestration:
    """Test the full generate_embeddings workflow using mocked helper methods."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    @pytest.fixture
    def sample_windows(self):
        return np.random.randn(5, 960, 5).astype(np.float32)

    def test_generate_embeddings_calls_all_steps(self, api_client, sample_windows):
        """Test that generate_embeddings calls upload, train, poll, and infer."""
        mock_embeddings = {str(i): np.random.randn(128).tolist() for i in range(5)}

        with patch.object(api_client, "_upload_dataset", return_value={"id": "ds_1"}) as mock_upload, \
             patch.object(api_client, "_train_model", return_value={"id": "model_1", "training_status": "PENDING"}) as mock_train, \
             patch.object(api_client, "_poll_model_status", return_value={"id": "model_1", "training_status": "COMPLETE"}) as mock_poll, \
             patch.object(api_client, "_infer", return_value=mock_embeddings) as mock_infer:

            embeddings = api_client.generate_embeddings(sample_windows)

            mock_upload.assert_called_once()
            mock_train.assert_called_once()
            mock_poll.assert_called_once_with("model_1", progress_callback=None)
            mock_infer.assert_called_once_with("model_1", "ds_1")
            assert embeddings.shape == (5, 128)

    def test_generate_embeddings_preserves_values(self, api_client, sample_windows):
        """Test that embedding values from inference are preserved."""
        expected = np.array([[0.1, 0.2, 0.3]] * 5, dtype=np.float32)
        mock_embeddings = {str(i): [0.1, 0.2, 0.3] for i in range(5)}

        with patch.object(api_client, "_upload_dataset", return_value={"id": "ds_1"}), \
             patch.object(api_client, "_train_model", return_value={"id": "m_1", "training_status": "PENDING"}), \
             patch.object(api_client, "_poll_model_status", return_value={"training_status": "COMPLETE"}), \
             patch.object(api_client, "_infer", return_value=mock_embeddings):

            embeddings = api_client.generate_embeddings(sample_windows)
            np.testing.assert_array_almost_equal(embeddings, expected)

    def test_empty_windows_returns_empty(self, api_client):
        """Test that empty windows return empty array without API calls."""
        empty = np.array([]).reshape(0, 960, 5)
        with patch.object(api_client, "_upload_dataset") as mock_upload:
            embeddings = api_client.generate_embeddings(empty)
            mock_upload.assert_not_called()
            assert embeddings.shape[0] == 0
            assert embeddings.ndim == 2


class TestUploadDataset:
    """Test dataset upload request formatting."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    def test_upload_sends_multipart(self, api_client):
        """Test that upload sends multipart form data."""
        with patch.object(api_client, "_make_request", return_value={"id": "ds_1"}) as mock:
            api_client._upload_dataset(b"col1,col2\n1,2\n", "test_ds")

            mock.assert_called_once()
            call_kwargs = mock.call_args
            assert call_kwargs[0][0] == "POST"
            assert call_kwargs[0][1] == "/api/datasets"
            assert call_kwargs[1]["data"] == {"name": "test_ds", "overwrite": "true"}
            assert "file" in call_kwargs[1]["files"]


class TestTrainModel:
    """Test model training request formatting."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    def test_train_sends_correct_params(self, api_client):
        """Test that train request has correct body and query params."""
        with patch.object(api_client, "_make_request",
                         return_value={"id": "m_1", "training_status": "PENDING"}) as mock:
            api_client._train_model("my_dataset", "my_model")

            mock.assert_called_once_with(
                "POST",
                "/api/models/embedding/train",
                data={"model_name": "my_model", "overwrite": True},
                params={"dataset_name": "my_dataset"},
                use_form=True
            )


class TestPollModelStatus:
    """Test training status polling behavior."""

    @pytest.fixture
    def api_client(self):
        client = APIClient(api_key="test_key_12345", poll_interval=0)
        return client

    def test_poll_returns_on_complete(self, api_client):
        """Test that polling returns when status is COMPLETE."""
        with patch.object(api_client, "_make_request",
                         return_value={"id": "m_1", "training_status": "COMPLETE"}):
            result = api_client._poll_model_status("m_1")
            assert result["training_status"] == "COMPLETE"

    def test_poll_retries_on_pending(self, api_client):
        """Test that polling retries when status is PENDING."""
        responses = [
            {"id": "m_1", "training_status": "PENDING"},
            {"id": "m_1", "training_status": "PENDING"},
            {"id": "m_1", "training_status": "COMPLETE"},
        ]
        with patch.object(api_client, "_make_request", side_effect=responses) as mock:
            result = api_client._poll_model_status("m_1")
            assert mock.call_count == 3
            assert result["training_status"] == "COMPLETE"

    def test_poll_raises_on_failure(self, api_client):
        """Test that polling raises on FAILED status."""
        with patch.object(api_client, "_make_request",
                         return_value={"id": "m_1", "training_status": "FAILED"}):
            with pytest.raises(WoodWideAPIError, match="training failed"):
                api_client._poll_model_status("m_1")

    def test_poll_raises_on_timeout(self, api_client):
        """Test that polling raises after timeout."""
        api_client.training_timeout = 0  # Immediate timeout
        with patch.object(api_client, "_make_request",
                         return_value={"id": "m_1", "training_status": "PENDING"}):
            with pytest.raises(TrainingTimeoutError, match="did not complete"):
                api_client._poll_model_status("m_1")


class TestHealthCheck:
    """Test health check with non-JSON response."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    def test_health_check_empty_body(self, api_client):
        """Test health check handles empty response body."""
        with patch.object(api_client, "_make_request", return_value=None):
            result = api_client.check_health()
            assert result["status"] == "healthy"

    def test_health_check_passes_expect_json_false(self, api_client):
        """Test that health check uses expect_json=False."""
        with patch.object(api_client, "_make_request", return_value=None) as mock:
            api_client.check_health()
            mock.assert_called_once_with("GET", "/health", expect_json=False)


class TestMakeRequestModes:
    """Test _make_request with different modes (JSON vs multipart)."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    def test_json_mode(self, api_client):
        """Test that data is sent as JSON when no files."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        with patch.object(api_client.session, "request", return_value=mock_response) as mock:
            api_client._make_request("POST", "/test", data={"key": "val"})
            call_kwargs = mock.call_args[1]
            assert call_kwargs["json"] == {"key": "val"}

    def test_multipart_mode(self, api_client):
        """Test that files are sent as multipart form data."""
        import io
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        files = {"file": ("test.csv", io.BytesIO(b"data"), "text/csv")}

        with patch.object(api_client.session, "request", return_value=mock_response) as mock:
            api_client._make_request("POST", "/test", data={"name": "x"}, files=files)
            call_kwargs = mock.call_args[1]
            assert call_kwargs["data"] == {"name": "x"}
            assert "files" in call_kwargs

    def test_form_mode(self, api_client):
        """Test that use_form sends data as form-encoded (not JSON)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        with patch.object(api_client.session, "request", return_value=mock_response) as mock:
            api_client._make_request("POST", "/test", data={"key": "val"}, use_form=True)
            call_kwargs = mock.call_args[1]
            assert call_kwargs["data"] == {"key": "val"}
            assert "json" not in call_kwargs

    def test_expect_json_false(self, api_client):
        """Test that expect_json=False returns None."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(api_client.session, "request", return_value=mock_response):
            result = api_client._make_request("GET", "/health", expect_json=False)
            assert result is None


class TestRequestHeaders:
    """Test that request headers are correctly set."""

    @pytest.fixture
    def api_client(self):
        return APIClient(api_key="test_key_12345")

    def test_authorization_header_sent(self, api_client):
        """Verify that Authorization header is correctly set."""
        assert "Authorization" in api_client.session.headers
        assert api_client.session.headers["Authorization"] == "Bearer test_key_12345"

    def test_no_content_type_on_session(self, api_client):
        """Verify Content-Type is NOT set at session level (set per-request)."""
        assert "Content-Type" not in api_client.session.headers

    def test_user_agent_header_sent(self, api_client):
        """Verify that User-Agent header is set."""
        assert "User-Agent" in api_client.session.headers
        assert "HealthSyncMonitor" in api_client.session.headers["User-Agent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
