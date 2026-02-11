# Wood Wide API Client Guide

## Overview

The `APIClient` class provides a robust interface for generating multivariate embeddings using the Wood Wide AI API. It handles authentication, request batching, retry logic, and error handling.

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
print(status)  # {'status': 'healthy', 'version': '1.0.0'}

# Generate embeddings for preprocessed windows
# windows shape: (n_windows, window_length, n_features)
embeddings = client.generate_embeddings(
    windows,
    batch_size=32,
    embedding_dim=128
)

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
    WoodWideAPIError
)

try:
    embeddings = client.generate_embeddings(windows)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, retry later")
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

### 4. Batching

Large datasets are automatically batched to avoid memory issues:

```python
# Process 1000 windows in batches of 32
embeddings = client.generate_embeddings(
    windows,  # shape: (1000, 960, 5)
    batch_size=32
)
# embeddings shape: (1000, 128)
```

### 5. Request Logging

All requests are logged for debugging:

```python
import logging

logging.basicConfig(level=logging.INFO)
# Now you'll see request details:
# INFO:api_client:Generating embeddings for 114 windows (batch_size=32)
# INFO:api_client:Processing batch 1/4
```

## Command-Line Usage

Generate embeddings from preprocessed data:

```bash
# Using real API
python3 generate_embeddings.py 1

# Using mock API (for testing)
python3 generate_embeddings.py 1 --mock

# Custom batch size and embedding dimension
python3 generate_embeddings.py 1 \
    --batch-size 64 \
    --embedding-dim 256
```

## API Methods

### `generate_embeddings(windows, batch_size, embedding_dim)`

Generate embeddings for multiple windows.

**Args:**
- `windows` (ndarray): Shape `(n_windows, window_length, n_features)`
- `batch_size` (int): Windows per API request (default: 32)
- `embedding_dim` (int): Target embedding dimension (default: API's default)

**Returns:**
- `ndarray`: Shape `(n_windows, embedding_dim)`

### `generate_single_embedding(window, embedding_dim)`

Generate embedding for a single window.

**Args:**
- `window` (ndarray): Shape `(window_length, n_features)`
- `embedding_dim` (int): Target embedding dimension

**Returns:**
- `ndarray`: Shape `(embedding_dim,)`

### `check_health()`

Check API health status.

**Returns:**
- `dict`: Status information

### `get_embedding_info()`

Get information about available models.

**Returns:**
- `dict`: Model capabilities and configurations

## Advanced Configuration

### Custom Base URL

```python
client = APIClient(
    api_key="your_key",
    base_url="https://custom.api.com/v1"
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

# Specific test
pytest tests/test_api_client.py::TestMockAPIClient::test_batching -v

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

## Performance Tips

1. **Batch Size:** Larger batches are faster but use more memory
   - Small datasets: `batch_size=64`
   - Large datasets: `batch_size=32`

2. **Connection Reuse:** Use context manager or reuse client instance

3. **Parallel Processing:** For multiple subjects, consider parallel requests:
   ```python
   from concurrent.futures import ThreadPoolExecutor

   def process_subject(subject_id):
       with APIClient() as client:
           data = load_subject(subject_id)
           return client.generate_embeddings(data['windows'])

   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_subject, range(1, 16))
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

**Solution:** The client automatically retries. For persistent issues, reduce `batch_size` or add delays between requests.

### Timeout Errors

```
WoodWideAPIError: Request timeout after 3 retries
```

**Solution:** Increase timeout:
```python
client = APIClient(timeout=120)  # 2 minutes
```

## API Response Format

The API returns embeddings in this format:

```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // Window 1 (128-dim)
    [0.234, -0.567, 0.890, ...],  // Window 2 (128-dim)
    ...
  ]
}
```

## Security Best Practices

1. ✅ **Never commit `.env` file** (already in `.gitignore`)
2. ✅ **Use environment variables** for API keys
3. ✅ **Rotate API keys** periodically
4. ✅ **Use HTTPS** (enforced by client)
5. ✅ **Monitor API usage** to detect anomalies

## Support

For API-related issues:
- Check Wood Wide API documentation
- Verify API key is valid
- Check API status at https://status.woodwide.ai
- Contact Wood Wide support for quota increases
