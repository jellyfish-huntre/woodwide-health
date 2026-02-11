"""
Tests for API response handling with mocked HTTP requests.

These tests verify that the APIClient correctly handles successful JSON
responses from the Wood Wide API endpoint.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.api_client import APIClient, WoodWideAPIError


class TestSuccessfulJSONResponse:
    """Test that APIClient handles successful JSON responses correctly."""

    @pytest.fixture
    def api_client(self):
        """Create APIClient with mocked API key."""
        return APIClient(api_key="test_key_12345")

    @pytest.fixture
    def sample_windows(self):
        """Create sample window data."""
        # 5 windows, 960 timesteps, 5 features
        return np.random.randn(5, 960, 5).astype(np.float32)

    @pytest.fixture
    def mock_embedding_response(self):
        """Create a mock successful embedding response."""
        # Generate 5 embeddings of dimension 128
        embeddings = np.random.randn(5, 128).astype(np.float32)
        # Normalize to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return {
            "embeddings": embeddings.tolist(),
            "model": "multivariate-v1",
            "status": "success"
        }

    def test_successful_embedding_response(self, api_client, sample_windows, mock_embedding_response):
        """
        Test that APIClient correctly handles a successful embedding response.

        This test verifies:
        1. Request is made with correct payload
        2. Response JSON is correctly parsed
        3. Embeddings are correctly extracted and converted to numpy array
        4. Shape matches expected output
        """
        # Mock the session.request method
        with patch.object(api_client.session, 'request') as mock_request:
            # Create mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_embedding_response
            mock_request.return_value = mock_response

            # Call generate_embeddings
            embeddings = api_client.generate_embeddings(
                sample_windows,
                batch_size=5,
                embedding_dim=128
            )

            # Verify request was made
            assert mock_request.called
            call_args = mock_request.call_args

            # Verify request parameters
            assert call_args[1]['method'] == 'POST'
            assert '/embeddings' in call_args[1]['url']
            assert call_args[1]['json'] is not None

            # Verify request payload structure
            request_data = call_args[1]['json']
            assert 'windows' in request_data
            assert 'config' in request_data
            assert request_data['config']['embedding_dim'] == 128

            # Verify response handling
            assert embeddings.shape == (5, 128)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.dtype in [np.float32, np.float64]

    def test_health_check_response(self, api_client):
        """Test that health check endpoint returns correct JSON."""
        mock_health_response = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 12345
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_health_response
            mock_request.return_value = mock_response

            # Call health check
            health = api_client.check_health()

            # Verify response
            assert health['status'] == 'healthy'
            assert health['version'] == '1.0.0'
            assert health['uptime'] == 12345
            assert mock_request.called

    def test_embedding_info_response(self, api_client):
        """Test that embedding info endpoint returns correct JSON."""
        mock_info_response = {
            "models": [
                {
                    "name": "multivariate-v1",
                    "embedding_dim": 128,
                    "max_sequence_length": 1024,
                    "description": "Multivariate health embeddings"
                },
                {
                    "name": "multivariate-v2",
                    "embedding_dim": 256,
                    "max_sequence_length": 2048,
                    "description": "Higher dimension embeddings"
                }
            ]
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_info_response
            mock_request.return_value = mock_response

            # Call get_embedding_info
            info = api_client.get_embedding_info()

            # Verify response
            assert 'models' in info
            assert len(info['models']) == 2
            assert info['models'][0]['name'] == 'multivariate-v1'
            assert info['models'][0]['embedding_dim'] == 128
            assert info['models'][1]['embedding_dim'] == 256

    def test_multiple_batches_successful_responses(self, api_client):
        """Test handling multiple successful batch responses."""
        # Create data that requires 3 batches (batch_size=10, total=25)
        windows = np.random.randn(25, 960, 5).astype(np.float32)

        # Track batch calls
        batch_responses = []

        def mock_request_side_effect(*args, **kwargs):
            """Generate appropriate response based on request."""
            mock_response = Mock()
            mock_response.status_code = 200

            # Check if this is an embedding request
            if 'json' in kwargs and 'windows' in kwargs['json']:
                n_windows = len(kwargs['json']['windows'])
                embeddings = np.random.randn(n_windows, 128).astype(np.float32)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                response_data = {
                    "embeddings": embeddings.tolist(),
                    "status": "success"
                }
                batch_responses.append(n_windows)
            else:
                response_data = {"status": "healthy"}

            mock_response.json.return_value = response_data
            return mock_response

        with patch.object(api_client.session, 'request', side_effect=mock_request_side_effect):
            embeddings = api_client.generate_embeddings(
                windows,
                batch_size=10,
                embedding_dim=128
            )

            # Verify all batches were processed
            # Should be 3 batches: 10 + 10 + 5 = 25
            assert len(batch_responses) == 3
            assert batch_responses == [10, 10, 5]

            # Verify final embeddings shape
            assert embeddings.shape == (25, 128)

    def test_json_parsing_with_nested_data(self, api_client):
        """Test that complex nested JSON is correctly parsed."""
        complex_response = {
            "embeddings": np.random.randn(5, 128).tolist(),
            "metadata": {
                "model_version": "1.2.3",
                "processing_info": {
                    "normalization": "L2",
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                "performance": {
                    "inference_time_ms": 42.5,
                    "batch_size": 5
                }
            },
            "warnings": []
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = complex_response
            mock_request.return_value = mock_response

            # Make request
            response = api_client._make_request(
                method='POST',
                endpoint='/embeddings',
                data={'windows': [[1, 2, 3]]}
            )

            # Verify nested data is accessible
            assert 'embeddings' in response
            assert 'metadata' in response
            assert response['metadata']['model_version'] == '1.2.3'
            assert response['metadata']['processing_info']['normalization'] == 'L2'
            assert response['metadata']['performance']['inference_time_ms'] == 42.5

    def test_response_headers_accessible(self, api_client):
        """Test that response headers are accessible if needed."""
        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'Content-Type': 'application/json',
                'X-Request-ID': 'req-123456',
                'X-RateLimit-Remaining': '100'
            }
            mock_response.json.return_value = {"status": "healthy"}
            mock_request.return_value = mock_response

            # Make request
            api_client.check_health()

            # Verify headers were present in response
            actual_response = mock_request.return_value
            assert actual_response.headers['Content-Type'] == 'application/json'
            assert actual_response.headers['X-Request-ID'] == 'req-123456'
            assert actual_response.headers['X-RateLimit-Remaining'] == '100'

    def test_empty_embeddings_list_handled(self, api_client):
        """Test that empty embeddings list is handled correctly."""
        empty_windows = np.array([]).reshape(0, 960, 5)

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [],
                "status": "success"
            }
            mock_request.return_value = mock_response

            embeddings = api_client.generate_embeddings(
                empty_windows,
                batch_size=32
            )

            # Verify empty array is returned with correct shape
            assert embeddings.shape[0] == 0
            assert embeddings.ndim == 2

    def test_response_with_additional_metadata(self, api_client, sample_windows):
        """Test that additional metadata in response doesn't break parsing."""
        response_with_extras = {
            "embeddings": np.random.randn(5, 128).tolist(),
            "model": "multivariate-v1",
            "status": "success",
            "request_id": "req-789",
            "processing_time_ms": 123.45,
            "extra_field_1": "ignored",
            "extra_field_2": {"nested": "also ignored"}
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response_with_extras
            mock_request.return_value = mock_response

            # Should still work despite extra fields
            embeddings = api_client.generate_embeddings(
                sample_windows,
                batch_size=5
            )

            assert embeddings.shape == (5, 128)

    def test_single_embedding_response(self, api_client):
        """Test generating a single embedding."""
        single_window = np.random.randn(960, 5).astype(np.float32)

        single_embedding_response = {
            "embeddings": [np.random.randn(128).tolist()],
            "status": "success"
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = single_embedding_response
            mock_request.return_value = mock_response

            embedding = api_client.generate_single_embedding(single_window)

            # Verify single embedding returned
            assert embedding.shape == (128,)
            assert isinstance(embedding, np.ndarray)


class TestRequestPayloadValidation:
    """Test that requests are correctly formatted."""

    @pytest.fixture
    def api_client(self):
        """Create APIClient with mocked API key."""
        return APIClient(api_key="test_key_12345")

    def test_request_payload_structure(self, api_client):
        """Verify that request payload has correct structure."""
        windows = np.random.randn(3, 960, 5).astype(np.float32)

        captured_payload = None

        def capture_payload(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get('json')

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": np.random.randn(3, 128).tolist()
            }
            return mock_response

        with patch.object(api_client.session, 'request', side_effect=capture_payload):
            api_client.generate_embeddings(windows, batch_size=3, embedding_dim=128)

            # Verify payload structure
            assert captured_payload is not None
            assert 'windows' in captured_payload
            assert 'config' in captured_payload
            assert captured_payload['config']['embedding_dim'] == 128
            assert captured_payload['config']['normalize'] is True

            # Verify windows are serializable (converted to lists)
            assert isinstance(captured_payload['windows'], list)
            assert len(captured_payload['windows']) == 3

    def test_authorization_header_sent(self, api_client):
        """Verify that Authorization header is correctly set."""
        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_request.return_value = mock_response

            api_client.check_health()

            # Verify session headers include authorization
            assert 'Authorization' in api_client.session.headers
            assert api_client.session.headers['Authorization'] == 'Bearer test_key_12345'

    def test_content_type_header_sent(self, api_client):
        """Verify that Content-Type header is set."""
        assert 'Content-Type' in api_client.session.headers
        assert api_client.session.headers['Content-Type'] == 'application/json'

    def test_user_agent_header_sent(self, api_client):
        """Verify that User-Agent header is set."""
        assert 'User-Agent' in api_client.session.headers
        assert 'HealthSyncMonitor' in api_client.session.headers['User-Agent']


class TestResponseIntegrity:
    """Test that responses maintain data integrity."""

    @pytest.fixture
    def api_client(self):
        """Create APIClient with mocked API key."""
        return APIClient(api_key="test_key_12345")

    def test_embedding_values_preserved(self, api_client):
        """Test that embedding values are preserved during parsing."""
        # Create specific embedding values to verify
        expected_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)

        response = {
            "embeddings": expected_embeddings.tolist()
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response
            mock_request.return_value = mock_response

            windows = np.random.randn(2, 960, 5).astype(np.float32)
            actual_embeddings = api_client.generate_embeddings(windows, batch_size=2)

            # Verify values are exactly preserved
            np.testing.assert_array_almost_equal(
                actual_embeddings,
                expected_embeddings,
                decimal=5
            )

    def test_float_precision_maintained(self, api_client):
        """Test that float precision is maintained."""
        # Use precise float values
        precise_embedding = [
            0.123456789,
            -0.987654321,
            0.555555555
        ]

        response = {
            "embeddings": [precise_embedding]
        }

        with patch.object(api_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response
            mock_request.return_value = mock_response

            windows = np.random.randn(1, 960, 5).astype(np.float32)
            embeddings = api_client.generate_embeddings(windows, batch_size=1)

            # Verify precision (within float32 limits)
            assert abs(embeddings[0, 0] - 0.123456789) < 1e-6
            assert abs(embeddings[0, 1] - (-0.987654321)) < 1e-6
            assert abs(embeddings[0, 2] - 0.555555555) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
