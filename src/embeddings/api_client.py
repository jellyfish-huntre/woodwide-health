"""
Wood Wide API Client for generating multivariate embeddings.

This client handles authentication, request formatting, batching,
and error handling for the Wood Wide API.
"""

import os
import time
import requests
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WoodWideAPIError(Exception):
    """Base exception for Wood Wide API errors."""
    pass


class AuthenticationError(WoodWideAPIError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(WoodWideAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class APIClient:
    """
    Client for Wood Wide AI multivariate embedding API.

    This client transforms time-series health data (PPG + accelerometer)
    into latent embeddings that capture contextual relationships between
    heart rate and physical activity.

    Example:
        >>> client = APIClient()
        >>> embeddings = client.generate_embeddings(windows)
        >>> # embeddings shape: (n_windows, embedding_dim)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Wood Wide API client.

        Args:
            api_key: API key for authentication. If None, reads from WOOD_WIDE_API_KEY env var.
            base_url: Base URL for API. If None, reads from WOOD_WIDE_API_URL env var.
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)

        Raises:
            AuthenticationError: If API key is not provided or found in environment
        """
        # Get API credentials from environment or parameters
        self.api_key = api_key or os.getenv("WOOD_WIDE_API_KEY")
        self.base_url = base_url or os.getenv(
            "WOOD_WIDE_API_URL",
            "https://api.woodwide.ai/v1"
        )

        if not self.api_key:
            raise AuthenticationError(
                "API key not found. Set WOOD_WIDE_API_KEY environment variable "
                "or provide api_key parameter."
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "HealthSyncMonitor/1.0"
        })

        logger.info(f"Initialized Wood Wide API client: {self.base_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/embeddings')
            data: Request body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            WoodWideAPIError: If request fails after retries
            RateLimitError: If rate limit is exceeded
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # Raise for error status codes
                response.raise_for_status()

                return response.json()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {e}")
                elif response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {e}")
                elif attempt == self.max_retries - 1:
                    raise WoodWideAPIError(f"HTTP error after {self.max_retries} retries: {e}")
                else:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            except requests.exceptions.Timeout as e:
                if attempt == self.max_retries - 1:
                    raise WoodWideAPIError(f"Request timeout after {self.max_retries} retries: {e}")
                else:
                    logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise WoodWideAPIError(f"Request failed: {e}")
                else:
                    logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)

    def generate_embeddings(
        self,
        windows: np.ndarray,
        batch_size: int = 32,
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for time-series windows.

        This is the main method for transforming preprocessed health data
        into multivariate embeddings that capture contextual relationships.

        Args:
            windows: Array of shape (n_windows, window_length, n_features)
                     Each window contains synchronized PPG and accelerometer data
            batch_size: Number of windows to process per API call (default: 32)
            embedding_dim: Target embedding dimension (default: API's default)

        Returns:
            Array of shape (n_windows, embedding_dim) containing embeddings

        Example:
            >>> # windows shape: (114, 960, 5)
            >>> embeddings = client.generate_embeddings(windows)
            >>> # embeddings shape: (114, 128)
        """
        n_windows = windows.shape[0]
        logger.info(f"Generating embeddings for {n_windows} windows (batch_size={batch_size})")

        all_embeddings = []

        # Process in batches
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            batch = windows[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(n_windows + batch_size - 1)//batch_size}")

            # Convert to list format for JSON serialization
            batch_data = {
                "windows": batch.tolist(),
                "config": {
                    "embedding_dim": embedding_dim,
                    "normalize": True
                }
            }

            # Make API request
            response = self._make_request(
                method="POST",
                endpoint="/embeddings",
                data=batch_data
            )

            # Extract embeddings from response
            batch_embeddings = np.array(response["embeddings"])
            all_embeddings.append(batch_embeddings)

            # Be nice to the API
            time.sleep(0.1)

        # Concatenate all batches
        if len(all_embeddings) == 0:
            # Handle empty input
            embedding_dim_actual = embedding_dim or 128  # Default dimension
            embeddings = np.empty((0, embedding_dim_actual), dtype=np.float32)
        else:
            embeddings = np.vstack(all_embeddings)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def generate_single_embedding(
        self,
        window: np.ndarray,
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single window.

        Convenience method for processing one window at a time.

        Args:
            window: Array of shape (window_length, n_features)
            embedding_dim: Target embedding dimension

        Returns:
            Array of shape (embedding_dim,) containing the embedding
        """
        # Add batch dimension
        windows = np.expand_dims(window, axis=0)
        embeddings = self.generate_embeddings(windows, batch_size=1, embedding_dim=embedding_dim)
        return embeddings[0]

    def check_health(self) -> Dict:
        """
        Check API health status.

        Returns:
            Dictionary with API status information

        Example:
            >>> status = client.check_health()
            >>> print(status['status'])  # 'healthy'
        """
        try:
            response = self._make_request("GET", "/health")
            logger.info("API health check successful")
            return response
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            raise

    def get_embedding_info(self) -> Dict:
        """
        Get information about available embedding models.

        Returns:
            Dictionary with model information (dimensions, capabilities, etc.)

        Example:
            >>> info = client.get_embedding_info()
            >>> print(info['models'])
        """
        response = self._make_request("GET", "/embeddings/info")
        return response

    def close(self):
        """Close the API session."""
        self.session.close()
        logger.info("API client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MockAPIClient(APIClient):
    """
    Mock API client for testing without real API calls.

    Generates random embeddings with the same shape as real API responses.
    Useful for development and testing.
    """

    def __init__(self, embedding_dim: int = 128, **kwargs):
        """
        Initialize mock client.

        Args:
            embedding_dim: Dimension of generated embeddings
            **kwargs: Ignored (for compatibility with APIClient)
        """
        # Don't call super().__init__() to avoid API key requirement
        self.embedding_dim = embedding_dim
        self.base_url = "mock://api.woodwide.ai/v1"
        logger.info(f"Initialized Mock API client (embedding_dim={embedding_dim})")

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, **kwargs) -> Dict:
        """Mock request that returns fake data."""
        logger.info(f"Mock request: {method} {endpoint}")

        if endpoint == "/health":
            return {"status": "healthy", "version": "1.0.0-mock"}

        elif endpoint == "/embeddings/info":
            return {
                "models": [
                    {
                        "name": "multivariate-v1",
                        "embedding_dim": self.embedding_dim,
                        "max_sequence_length": 1024
                    }
                ]
            }

        elif endpoint == "/embeddings" and data:
            # Generate deterministic embeddings based on input
            windows = np.array(data["windows"])
            n_windows = windows.shape[0]

            embeddings = []
            for window in windows:
                # Create deterministic embedding based on window content
                # Use hash of window data as seed for reproducibility
                window_hash = hash(window.tobytes()) % (2**31)
                rng = np.random.RandomState(window_hash)
                embedding = rng.randn(self.embedding_dim).astype(np.float32)

                # Normalize to unit length (common for embeddings)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            embeddings = np.array(embeddings)
            return {"embeddings": embeddings.tolist()}

        else:
            raise WoodWideAPIError(f"Mock endpoint not implemented: {endpoint}")

    def close(self):
        """Mock close (no-op)."""
        logger.info("Mock API client session closed")
