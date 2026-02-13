# Wood Wide API Client Guide

## Overview

The `APIClient` class provides a robust interface for generating multivariate embeddings using the Wood Wide AI API. It handles authentication, the async training workflow, retry logic, and error handling.

The API uses a multi-step workflow: upload CSV dataset, train an embedding model, poll for training completion, then run inference to retrieve embeddings.

## Setup

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
WOOD_WIDE_API_KEY=your_actual_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Basic Usage

### Using the Real API

```python
from src.embeddings.api_client import APIClient
import numpy as np

# Initialize client (reads API key from environment)
client = APIClient()

# Check API health
status = client.check_health()
print(status)  # {'status': 'healthy'}

# Generate embeddings for preprocessed windows
# windows shape: (n_windows, window_length, n_features)
# Internally: uploads CSV -> trains model -> polls -> runs inference
embeddings = client.generate_embeddings(windows)

# Close the session
client.close()
```

### Using the Mock API (for Testing)

```python
from src.embeddings.api_client import MockAPIClient

# No API key required for mock
client = MockAPIClient(embedding_dim=128)

# Same interface as real client
embeddings = client.generate_embeddings(windows)
```

### Context Manager (Recommended)

```python
with APIClient() as client:
    embeddings = client.generate_embeddings(windows)
    # Session automatically closed
```

## API Workflow

The `generate_embeddings()` method orchestrates a multi-step workflow:

1. **Convert to CSV** - Flattens each window `(window_length, n_features)` into a single row
2. **Upload dataset** - `POST /api/datasets` with multipart CSV file
3. **Train model** - `POST /api/models/embedding/train` starts training
4. **Poll status** - `GET /api/models/{id}` until `training_status: "COMPLETE"`
5. **Run inference** - `POST /api/models/embedding/{id}/infer` returns embeddings
6. **Parse response** - Converts `{"0": [vec], "1": [vec], ...}` into numpy array

All of this is handled automatically behind the `generate_embeddings()` interface.

## API Client Features

### 1. Authentication

- **API Key:** Required for real API, read from `WOOD_WIDE_API_KEY` environment variable
- **Headers:** Automatically includes Authorization bearer token
- **Security:** Never hardcodes credentials

### 2. Error Handling

The client handles various error scenarios:

```python
from src.embeddings.api_client import (
    AuthenticationError,
    RateLimitError,
    TrainingTimeoutError,
    SSLConnectionError,
    WoodWideAPIError
)

try:
    embeddings = client.generate_embeddings(windows)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, retry later")
except TrainingTimeoutError:
    print("Model training took too long")
except SSLConnectionError:
    print("SSL handshake failed - try setting WOOD_WIDE_SSL_CIPHERS")
except WoodWideAPIError as e:
    print(f"API error: {e}")
```

### 3. Retry Logic

- **Max Retries:** 3 attempts (configurable)
- **Backoff:** Exponential backoff (1s, 2s, 4s)
- **Rate Limiting:** Automatically respects `Retry-After` header

```python
client = APIClient(
    max_retries=5,
    retry_delay=2.0,  # Initial delay
    timeout=60  # Request timeout
)
```

### 4. Training Timeout

Model training is polled until complete, with configurable timeout:

```python
client = APIClient(
    training_timeout=600,  # Max 10 minutes for training (default)
    poll_interval=5.0      # Check every 5 seconds (default)
)
```

### 5. Local Disk Caching

Cache embeddings to avoid re-running the API workflow on repeated calls:

```python
client = APIClient(cache_dir="data/embedding_cache")

embeddings = client.generate_embeddings(windows)   # Calls API, caches result
embeddings = client.generate_embeddings(windows)   # Loaded from disk cache

# Force a fresh API call, bypassing cache
embeddings = client.generate_embeddings(windows, force_regenerate=True)
```

Cache files are stored as compressed `.npz` files keyed by a SHA-256 hash of the input data.

### 6. Cleanup on Error

Uploaded datasets are automatically deleted if training or inference fails, preventing resource leaks on the API server:

```python
client = APIClient()
# If training fails, the uploaded dataset is cleaned up automatically
embeddings = client.generate_embeddings(windows)

# Disable cleanup if you want to keep the dataset for debugging
embeddings = client.generate_embeddings(windows, cleanup_on_error=False)
```

### 7. Server-Side Model Reuse

Before training a new model, the client checks `list_models()` for an existing model with the same name and COMPLETE status. If found, training is skipped entirely and the existing model is used for inference. Model names are deterministic (based on a hash of the input data), so repeated calls with the same data automatically reuse the trained model.

### 8. Request Logging

All requests are logged for debugging:

```python
import logging

logging.basicConfig(level=logging.INFO)
# Now you'll see request details:
# INFO:api_client:Generating embeddings for 114 windows
# INFO:api_client:Uploading dataset 'health_windows_a1b2c3d4'...
# INFO:api_client:Waiting for training to complete...
# INFO:api_client:Model model_id training complete
# INFO:api_client:Running inference...
```

## Command-Line Usage

Generate embeddings from preprocessed data:

```bash
# Using real API
python3 generate_embeddings.py 1

# Using mock API (for testing)
python3 generate_embeddings.py 1 --mock

# Custom dataset and model names
python3 generate_embeddings.py 1 \
    --dataset-name my_health_data \
    --model-name my_embed_model
```

## API Methods

### `generate_embeddings(windows, dataset_name, model_name)`

Generate embeddings for multiple windows via the upload-train-infer workflow.

**Args:**
- `windows` (ndarray): Shape `(n_windows, window_length, n_features)`
- `dataset_name` (str, optional): Name for uploaded dataset (auto-generated if None)
- `model_name` (str, optional): Name for trained model (auto-generated if None)
- `batch_size` (int): Kept for backward compatibility (ignored)
- `embedding_dim` (int): Kept for backward compatibility (ignored)
- `cleanup_on_error` (bool): Delete uploaded dataset on failure (default: True)
- `force_regenerate` (bool): Bypass local disk cache (default: False)

**Returns:**
- `ndarray`: Shape `(n_windows, embedding_dim)`

### `generate_single_embedding(window)`

Generate embedding for a single window. Triggers a full upload-train-infer cycle.

**Args:**
- `window` (ndarray): Shape `(window_length, n_features)`

**Returns:**
- `ndarray`: Shape `(embedding_dim,)`

### `check_health()`

Check API health status.

**Returns:**
- `dict`: `{"status": "healthy"}` on success

### `get_user_info()`

Get authenticated user info including credits and resource IDs.

**Returns:**
- `dict`: With `wwai_credits`, `dataset_ids`, `model_ids`, etc.

### `list_datasets()`

List all uploaded datasets.

### `list_models()`

List all trained models.

## Advanced Configuration

### Custom Base URL

```python
client = APIClient(
    api_key="your_key",
    base_url="https://custom.api.com"
)
```

### Multiple API Keys (Different Projects)

```python
# Project A
client_a = APIClient(api_key="project_a_key")

# Project B
client_b = APIClient(api_key="project_b_key")
```

### Connection Pooling

The client uses session-based connection pooling for efficiency:

```python
# Reuse the same client for multiple requests
with APIClient() as client:
    for subject_id in range(1, 16):
        data = load_subject(subject_id)
        embeddings = client.generate_embeddings(data['windows'])
        save_embeddings(embeddings, subject_id)
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/test_api_client.py -v

# Response handling tests
pytest tests/test_api_response_handling.py -v

# With coverage
pytest tests/test_api_client.py --cov=src.embeddings
```

### Mock Client for Development

Use `MockAPIClient` during development to avoid API costs:

```python
if os.getenv("ENVIRONMENT") == "development":
    client = MockAPIClient(embedding_dim=128)
else:
    client = APIClient()

# Same code works with both
embeddings = client.generate_embeddings(windows)
```

## Troubleshooting

### Authentication Errors

```
AuthenticationError: API key not found
```

**Solution:** Set `WOOD_WIDE_API_KEY` in `.env` file

### Rate Limiting

```
RateLimitError: Rate limit exceeded
```

**Solution:** The client automatically retries. For persistent issues, add delays between requests.

### SSL Handshake Failures

```
SSLConnectionError: SSL handshake failed connecting to https://beta.woodwide.ai/health
```

**Cause:** OpenSSL 3.0+ uses stricter default cipher settings that may be incompatible with the API server.

**Solution 1:** Set a broader cipher configuration in your `.env` file:
```bash
WOOD_WIDE_SSL_CIPHERS=DEFAULT:@SECLEVEL=0
```

**Solution 2:** Pass cipher configuration programmatically:
```python
client = APIClient(ssl_ciphers="DEFAULT:@SECLEVEL=0")
```

### Training Timeout

```
TrainingTimeoutError: Model training did not complete within 600s
```

**Solution:** Increase the training timeout:
```python
client = APIClient(training_timeout=1200)  # 20 minutes
```

### Timeout Errors

```
WoodWideAPIError: Request timeout after 3 retries
```

**Solution:** Increase timeout:
```python
client = APIClient(timeout=120)  # 2 minutes
```

## API Response Format

The inference endpoint returns embeddings indexed by row:

```json
{
  "0": [0.123, -0.456, 0.789, ...],
  "1": [0.234, -0.567, 0.890, ...],
  ...
}
```

The client automatically converts this into a numpy array of shape `(n_windows, embedding_dim)`.

## Security Best Practices

1. **Never commit `.env` file** (already in `.gitignore`)
2. **Use environment variables** for API keys
3. **Rotate API keys** periodically
4. **Use HTTPS** (enforced by client)
5. **Monitor API usage** to detect anomalies

## Support

For API-related issues:
- Check Wood Wide API documentation at https://docs.woodwide.ai
- Verify API key is valid
- Check API status at https://status.woodwide.ai
- Contact Wood Wide support for quota increases
