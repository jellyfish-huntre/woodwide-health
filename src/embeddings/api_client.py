"""
Wood Wide API Client for generating multivariate embeddings.

This client handles authentication, request formatting, the async
training workflow, and error handling for the Wood Wide API.

Workflow: upload CSV dataset → train embedding model → poll for
completion → run inference → return embeddings.
"""

import csv
import hashlib
import io
import os
import ssl
import time
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
import numpy as np
from typing import Callable, Dict, Optional
from dotenv import load_dotenv
from urllib3.util.ssl_ import create_urllib3_context
import logging

load_dotenv()
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


class SSLConnectionError(WoodWideAPIError):
    """Raised when SSL/TLS handshake or connection fails."""
    pass


class InsufficientCreditsError(WoodWideAPIError):
    """Raised when the API key has run out of credits (HTTP 402)."""
    pass


class TrainingTimeoutError(WoodWideAPIError):
    """Raised when model training does not complete within the timeout."""
    pass


class _SSLAdapter(HTTPAdapter):
    """HTTPAdapter that applies a custom SSL context to all connections."""

    def __init__(self, ssl_context=None, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


class APIClient:
    """
    Client for Wood Wide AI multivariate embedding API.

    This client transforms time-series health data (PPG + accelerometer)
    into latent embeddings that capture contextual relationships between
    heart rate and physical activity.

    The workflow is: upload CSV → train embedding model → poll until
    complete → run inference → return embeddings as numpy array.

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
        retry_delay: float = 1.0,
        ssl_ciphers: Optional[str] = None,
        training_timeout: int = 600,
        poll_interval: float = 5.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Wood Wide API client.

        Args:
            api_key: API key for authentication. If None, reads from WOOD_WIDE_API_KEY env var.
            base_url: Base URL for API. If None, reads from WOOD_WIDE_API_URL env var.
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            ssl_ciphers: SSL cipher string override. If None, reads from WOOD_WIDE_SSL_CIPHERS env var.
            training_timeout: Max seconds to wait for model training (default: 600)
            poll_interval: Seconds between training status polls (default: 5.0)
            cache_dir: Directory for local embedding cache. If None, caching is disabled.

        Raises:
            AuthenticationError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("WOOD_WIDE_API_KEY")
        self.base_url = base_url or os.getenv(
            "WOOD_WIDE_API_URL",
            "https://beta.woodwide.ai"
        )

        if not self.api_key:
            raise AuthenticationError(
                "API key not found. Set WOOD_WIDE_API_KEY environment variable "
                "or provide api_key parameter."
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ssl_ciphers = ssl_ciphers or os.getenv("WOOD_WIDE_SSL_CIPHERS")
        self.training_timeout = training_timeout
        self.poll_interval = poll_interval
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache directory: {self.cache_dir}")

        # Session for connection pooling (no Content-Type — set per-request)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "HealthSyncMonitor/1.0"
        })

        # Mount SSL adapter for OpenSSL 3.0+ compatibility
        ssl_context = self._create_ssl_context(ciphers=self.ssl_ciphers)
        self.session.mount("https://", _SSLAdapter(ssl_context=ssl_context))

        logger.info(f"Initialized Wood Wide API client: {self.base_url}")

    @staticmethod
    def _create_ssl_context(ciphers: Optional[str] = None) -> ssl.SSLContext:
        """Create an SSL context with broader cipher suite support for OpenSSL 3.0+."""
        ctx = create_urllib3_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.set_ciphers(ciphers or "DEFAULT:@SECLEVEL=0")
        return ctx

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
        expect_json: bool = True,
        use_form: bool = False
    ) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/api/datasets')
            data: Request body data (sent as JSON by default)
            params: Query parameters
            files: Multipart file uploads (when provided, data is sent as form fields)
            expect_json: If False, don't attempt to parse response as JSON
            use_form: If True, send data as form-encoded instead of JSON

        Returns:
            Response JSON data, or None if expect_json is False

        Raises:
            WoodWideAPIError: If request fails after retries
            RateLimitError: If rate limit is exceeded
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                # Reset file streams for retries (BytesIO gets consumed on each request)
                if files:
                    for file_tuple in files.values():
                        if hasattr(file_tuple[1], 'seek'):
                            file_tuple[1].seek(0)

                # When files are provided, use multipart form data
                if files:
                    response = self.session.request(
                        method=method,
                        url=url,
                        data=data,
                        files=files,
                        params=params,
                        timeout=self.timeout
                    )
                elif use_form:
                    response = self.session.request(
                        method=method,
                        url=url,
                        data=data,
                        params=params,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        timeout=self.timeout
                    )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                if not expect_json:
                    return None
                return response.json()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {e}")
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        "API key has run out of credits. "
                        "Please add credits or use a different API key."
                    )
                elif response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {e}")
                elif attempt == self.max_retries - 1:
                    raise WoodWideAPIError(f"HTTP error after {self.max_retries} retries: {e}")
                else:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))

            except requests.exceptions.SSLError as e:
                error_msg = str(e)
                if "HANDSHAKE_FAILURE" in error_msg or "handshake" in error_msg.lower():
                    raise SSLConnectionError(
                        f"SSL handshake failed connecting to {url}. "
                        f"This may be caused by cipher suite incompatibility with OpenSSL 3.0+. "
                        f"Try setting WOOD_WIDE_SSL_CIPHERS='DEFAULT:@SECLEVEL=0' in your .env file. "
                        f"Original error: {e}"
                    )
                if attempt == self.max_retries - 1:
                    raise SSLConnectionError(f"SSL error after {self.max_retries} retries: {e}")
                else:
                    logger.warning(f"SSL error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))

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

    # ---- Dataset & model workflow helpers ----

    # Maximum number of CSV columns the API can train on reliably.
    # Beyond this, internal KMeans/PCA steps may produce NaN.
    MAX_FLAT_COLS = 500

    @staticmethod
    def _extract_summary_features(windows: np.ndarray) -> np.ndarray:
        """Extract per-window summary statistics for API upload.

        When raw windows are too wide (e.g. 960 samples x 5 features = 4800
        columns), the API's internal training fails.  This converts each
        window into a compact feature vector of summary statistics that
        preserves the key signal characteristics.

        Args:
            windows: Shape (n_windows, window_length, n_features)

        Returns:
            Feature matrix of shape (n_windows, n_summary_features)
        """
        n_windows, window_length, n_features = windows.shape

        # Compute all 7 statistics along the time axis (axis=1)
        # Each result has shape (n_windows, n_features)
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        mins = np.min(windows, axis=1)
        maxs = np.max(windows, axis=1)
        medians = np.median(windows, axis=1)
        q25 = np.percentile(windows, 25, axis=1)
        q75 = np.percentile(windows, 75, axis=1)

        # Stack into (n_windows, 7, n_features), transpose to
        # (n_windows, n_features, 7), reshape to (n_windows, n_features * 7).
        # This produces interleaved order matching the original for-loop:
        # [mean_f0, std_f0, min_f0, max_f0, med_f0, q25_f0, q75_f0, mean_f1, ...]
        stats = np.stack([means, stds, mins, maxs, medians, q25, q75], axis=1)
        stats = np.transpose(stats, (0, 2, 1))
        features = stats.reshape(n_windows, -1)

        # Cross-feature: PPG–accelerometer-magnitude correlation
        if n_features >= 5:
            ppg = windows[:, :, 0]
            acc_mag = windows[:, :, 4]
            ppg_centered = ppg - ppg.mean(axis=1, keepdims=True)
            acc_centered = acc_mag - acc_mag.mean(axis=1, keepdims=True)
            numerator = (ppg_centered * acc_centered).sum(axis=1)
            denom = np.sqrt(
                (ppg_centered ** 2).sum(axis=1) * (acc_centered ** 2).sum(axis=1)
            )
            corr = np.where(denom > 0, numerator / denom, 0.0)
            features = np.column_stack([features, corr])

        features = features.astype(np.float32)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _windows_to_csv_bytes(windows: np.ndarray) -> bytes:
        """Convert windows array to CSV bytes for upload.

        Each window (window_length x n_features) is flattened to a single row.
        This way each row in the dataset produces one embedding vector.

        Args:
            windows: Shape (n_windows, window_length, n_features) or (n_windows, n_features)

        Returns:
            CSV file content as bytes
        """
        n_windows = windows.shape[0]
        flat = windows.reshape(n_windows, -1)
        n_cols = flat.shape[1]

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([f"f_{i}" for i in range(n_cols)])
        for row in flat:
            writer.writerow(row.tolist())

        return buf.getvalue().encode("utf-8")

    def _upload_dataset(self, csv_bytes: bytes, name: str, overwrite: bool = True) -> Dict:
        """Upload CSV data as a dataset.

        POST /api/datasets with multipart form: file + name + overwrite fields.

        Args:
            csv_bytes: CSV file content
            name: Dataset name
            overwrite: If True, overwrite existing dataset with same name

        Returns:
            Response dict with 'id', 'name', 'size', 'schema'
        """
        files = {"file": (f"{name}.csv", io.BytesIO(csv_bytes), "text/csv")}
        data = {"name": name, "overwrite": str(overwrite).lower()}
        return self._make_request("POST", "/api/datasets", data=data, files=files)

    def _delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset by ID.

        DELETE /api/datasets/{dataset_id}
        """
        self._make_request("DELETE", f"/api/datasets/{dataset_id}", expect_json=False)

    def _train_model(self, dataset_name: str, model_name: str, overwrite: bool = True) -> Dict:
        """Start training an embedding model.

        POST /api/models/embedding/train?dataset_name=<name>

        Returns:
            Response dict with 'id', 'training_status': 'PENDING'
        """
        return self._make_request(
            "POST",
            "/api/models/embedding/train",
            data={"model_name": model_name, "overwrite": overwrite},
            params={"dataset_name": dataset_name},
            use_form=True
        )

    def _poll_model_status(
        self,
        model_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Poll model status until training completes.

        GET /api/models/{model_id} repeatedly until COMPLETE or timeout.

        Args:
            model_id: Model ID to poll
            progress_callback: Optional callback(step, message) for progress updates

        Returns:
            Model info dict with training_status == 'COMPLETE'

        Raises:
            TrainingTimeoutError: If training does not complete in time
            WoodWideAPIError: If training fails
        """
        start = time.time()
        while True:
            model_info = self._make_request("GET", f"/api/models/{model_id}")
            status = model_info.get("training_status", "UNKNOWN")

            elapsed = time.time() - start
            if status == "COMPLETE":
                msg = f"Model {model_id} training complete ({elapsed:.0f}s)"
                logger.info(msg)
                if progress_callback:
                    progress_callback("poll_done", msg)
                return model_info
            elif status in ("FAILED", "ERROR"):
                raise WoodWideAPIError(f"Model training failed with status: {status}")

            if elapsed > self.training_timeout:
                raise TrainingTimeoutError(
                    f"Model training did not complete within {self.training_timeout}s "
                    f"(last status: {status})"
                )

            msg = f"Model {model_id} status: {status} (elapsed: {elapsed:.0f}s)"
            logger.info(msg)
            if progress_callback:
                progress_callback("poll_status", msg)
            time.sleep(self.poll_interval)

    def _infer(self, model_id: str, dataset_id: str) -> Dict:
        """Run inference to get embeddings.

        POST /api/models/embedding/{model_id}/infer?dataset_id=<id>

        Returns:
            Dict mapping row_index (str) -> embedding vector (list of floats)
        """
        return self._make_request(
            "POST",
            f"/api/models/embedding/{model_id}/infer",
            params={"dataset_id": dataset_id}
        )

    # ---- Public API ----

    def generate_embeddings(
        self,
        windows: np.ndarray,
        batch_size: int = 32,
        embedding_dim: Optional[int] = None,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        cleanup_on_error: bool = True,
        force_regenerate: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for time-series windows.

        Orchestrates the full workflow: convert to CSV → upload dataset →
        train embedding model → poll until complete → run inference.

        Includes automatic cleanup of uploaded datasets on failure, server-side
        model reuse (skips training if a matching model already exists), and
        optional local disk caching via ``cache_dir``.

        Args:
            windows: Array of shape (n_windows, window_length, n_features)
            batch_size: Ignored (kept for backward compatibility)
            embedding_dim: Ignored (kept for backward compatibility)
            dataset_name: Name for the uploaded dataset (auto-generated if None)
            model_name: Name for the trained model (auto-generated if None)
            progress_callback: Optional callback(step, message) for progress updates.
                Steps: csv, upload, upload_done, reuse, train, train_done, poll,
                poll_status, poll_done, infer, cache_hit, done.
            cleanup_on_error: If True, delete the uploaded dataset when
                training or inference fails (default: True).
            force_regenerate: If True, bypass the local disk cache and
                regenerate embeddings from the API (default: False).

        Returns:
            Array of shape (n_windows, embedding_dim) containing embeddings
        """
        def _notify(step: str, message: str):
            logger.info(message)
            if progress_callback:
                progress_callback(step, message)

        n_windows = windows.shape[0]
        if n_windows == 0:
            return np.empty((0, 0), dtype=np.float32)

        _notify("start", f"Generating embeddings for {n_windows} windows")

        # Generate deterministic names and cache key from data content
        data_hash = hashlib.md5(windows.tobytes()[:1024]).hexdigest()[:8]
        cache_key = hashlib.sha256(windows.tobytes()).hexdigest()[:16]
        dataset_name = dataset_name or f"health_windows_{data_hash}"
        model_name = model_name or f"health_embed_{data_hash}"

        if self.cache_dir and not force_regenerate:
            cache_file = self.cache_dir / f"embeddings_{cache_key}.npz"
            if cache_file.exists():
                _notify("cache_hit", f"Loading cached embeddings from {cache_file}")
                cached = np.load(cache_file)
                embeddings = cached["embeddings"]
                _notify("done", f"Loaded cached embeddings shape: {embeddings.shape}")
                return embeddings

        flat_cols = windows.shape[1] * windows.shape[2] if windows.ndim == 3 else windows.shape[1]
        if flat_cols > self.MAX_FLAT_COLS:
            _notify("csv", f"Extracting summary features from {n_windows} windows ({flat_cols} cols → compact)...")
            summary = self._extract_summary_features(windows)
            csv_bytes = self._windows_to_csv_bytes(summary)
        else:
            _notify("csv", f"Converting {n_windows} windows to CSV format...")
            csv_bytes = self._windows_to_csv_bytes(windows)

        _notify("upload", f"Uploading dataset '{dataset_name}'...")
        dataset_info = self._upload_dataset(csv_bytes, dataset_name)
        dataset_id = dataset_info["id"]
        _notify("upload_done", f"Dataset uploaded: id={dataset_id}")

        try:
            model_id = None
            try:
                existing_models = self.list_models()
                for model in existing_models:
                    if (model.get("model_name") == model_name and
                            model.get("training_status") == "COMPLETE"):
                        model_id = model["id"]
                        _notify("reuse", f"Found existing model '{model_name}' (id={model_id}), skipping training")
                        break
            except Exception as e:
                logger.debug(f"Could not check for existing models: {e}")

            if model_id is None:
                _notify("train", f"Starting model training '{model_name}'...")
                train_response = self._train_model(dataset_name, model_name)
                model_id = train_response["id"]
                _notify("train_done", f"Training started: model_id={model_id}")

                _notify("poll", "Waiting for training to complete...")
                self._poll_model_status(model_id, progress_callback=progress_callback)

            _notify("infer", "Running inference...")
            infer_response = self._infer(model_id, dataset_id)

            # Response format: {"embeddings": {"0": [vec], "1": [vec], ...}}
            embeddings_dict = infer_response.get("embeddings", infer_response)
            embeddings_list = []
            for i in range(n_windows):
                vec = embeddings_dict.get(str(i))
                if vec is None:
                    raise WoodWideAPIError(f"Missing embedding for window {i} in inference response")
                embeddings_list.append(vec)

            embeddings = np.array(embeddings_list, dtype=np.float32)

            if self.cache_dir:
                cache_file = self.cache_dir / f"embeddings_{cache_key}.npz"
                np.savez_compressed(cache_file, embeddings=embeddings)
                logger.info(f"Cached embeddings to {cache_file}")

            _notify("done", f"Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception:
            if cleanup_on_error:
                try:
                    logger.info(f"Cleaning up dataset {dataset_id} after failure...")
                    self._delete_dataset(dataset_id)
                    logger.info(f"Dataset {dataset_id} deleted successfully")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up dataset {dataset_id}: {cleanup_err}")
            raise

    def generate_single_embedding(
        self,
        window: np.ndarray,
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single window.

        WARNING: This triggers a full upload-train-infer cycle for one window.
        Prefer generate_embeddings() with multiple windows for efficiency.

        Args:
            window: Array of shape (window_length, n_features)
            embedding_dim: Ignored (kept for backward compatibility)

        Returns:
            Array of shape (embedding_dim,) containing the embedding
        """
        windows = np.expand_dims(window, axis=0)
        embeddings = self.generate_embeddings(windows, embedding_dim=embedding_dim)
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
            self._make_request("GET", "/health", expect_json=False)
            logger.info("API health check successful")
            return {"status": "healthy"}
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            raise

    def get_user_info(self) -> Dict:
        """
        Get authenticated user info including credits and resource IDs.

        Returns:
            Dictionary with wwai_credits, dataset_ids, model_ids, etc.

        Example:
            >>> info = client.get_user_info()
            >>> print(info['wwai_credits'])
        """
        return self._make_request("GET", "/auth/me")

    def list_datasets(self) -> list:
        """List all datasets. GET /api/datasets."""
        return self._make_request("GET", "/api/datasets")

    def list_models(self) -> list:
        """List all models. GET /api/models."""
        return self._make_request("GET", "/api/models")

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

    Simulates the full upload → train → infer workflow with deterministic
    embeddings. Useful for development and testing.
    """

    def __init__(self, embedding_dim: int = 128, cache_dir: Optional[str] = None, **kwargs):
        """
        Initialize mock client.

        Args:
            embedding_dim: Dimension of generated embeddings
            cache_dir: Directory for local embedding cache. If None, caching is disabled.
            **kwargs: Ignored (for compatibility with APIClient)
        """
        # Don't call super().__init__() to avoid API key requirement
        self.embedding_dim = embedding_dim
        self.base_url = "mock://beta.woodwide.ai"
        self.training_timeout = 600
        self.poll_interval = 0  # No delay in mock
        self._mock_datasets = {}
        self._mock_models = {}
        self._dataset_counter = 0
        self._model_counter = 0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Mock API client (embedding_dim={embedding_dim})")

    def _make_request(self, method: str, endpoint: str, data=None,
                      params=None, files=None, expect_json=True,
                      use_form=False):
        """Mock request that simulates all API endpoints."""
        logger.info(f"Mock request: {method} {endpoint}")

        # GET /health
        if endpoint == "/health" and method == "GET":
            return None

        # GET /auth/me
        if endpoint == "/auth/me" and method == "GET":
            return {
                "identity_type": "api",
                "username": "mock_user",
                "dataset_ids": list(self._mock_datasets.keys()),
                "model_ids": list(self._mock_models.keys()),
                "created_at": None
            }

        # POST /api/datasets
        if endpoint == "/api/datasets" and method == "POST":
            self._dataset_counter += 1
            ds_id = f"ds_mock_{self._dataset_counter}"
            name = data.get("name", "unnamed") if data else "unnamed"
            csv_content = b""
            if files and "file" in files:
                file_tuple = files["file"]
                csv_content = file_tuple[1].read()
                file_tuple[1].seek(0)
            self._mock_datasets[ds_id] = {"csv": csv_content, "name": name}
            return {"id": ds_id, "name": name, "size": len(csv_content), "schema": {}}

        # DELETE /api/datasets/{dataset_id}
        if endpoint.startswith("/api/datasets/") and method == "DELETE":
            ds_id = endpoint.split("/")[-1]
            self._mock_datasets.pop(ds_id, None)
            return None

        # GET /api/datasets
        if endpoint == "/api/datasets" and method == "GET":
            return [{"id": k, "name": v["name"]} for k, v in self._mock_datasets.items()]

        # POST /api/models/embedding/train
        if endpoint == "/api/models/embedding/train" and method == "POST":
            self._model_counter += 1
            model_id = f"model_mock_{self._model_counter}"
            model_name_val = data.get("model_name", "unnamed") if data else "unnamed"
            self._mock_models[model_id] = {
                "training_status": "COMPLETE",
                "model_name": model_name_val
            }
            return {"id": model_id, "training_status": "PENDING"}

        # GET /api/models/{model_id}
        if endpoint.startswith("/api/models/") and method == "GET" and "/infer" not in endpoint:
            model_id = endpoint.split("/")[-1]
            info = self._mock_models.get(model_id, {"training_status": "COMPLETE"})
            return {"id": model_id, **info}

        # POST /api/models/embedding/{model_id}/infer
        if "/infer" in endpoint and method == "POST":
            dataset_id = params.get("dataset_id") if params else None
            csv_content = self._mock_datasets.get(dataset_id, {}).get("csv", b"")

            # Count rows (subtract 1 for header)
            n_rows = max(csv_content.count(b"\n") - 1, 0) if csv_content else 0

            # Generate deterministic embeddings seeded on CSV content
            csv_hash = hash(csv_content) % (2**31)
            embeddings = {}
            for i in range(n_rows):
                rng = np.random.RandomState((csv_hash + i) % (2**31))
                vec = rng.randn(self.embedding_dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                embeddings[str(i)] = vec.tolist()
            return {"embeddings": embeddings}

        # GET /api/models
        if endpoint == "/api/models" and method == "GET":
            return [{"id": k, **v} for k, v in self._mock_models.items()]

        raise WoodWideAPIError(f"Mock endpoint not implemented: {method} {endpoint}")

    def close(self):
        """Mock close (no-op)."""
        logger.info("Mock API client session closed")
