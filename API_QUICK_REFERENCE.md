# Wood Wide API Client - Quick Reference

## Setup (One-time)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and add your API key
echo "WOOD_WIDE_API_KEY=your_key_here" >> .env

# 3. Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Python API

```python
from src.embeddings.api_client import APIClient

# Initialize
with APIClient() as client:
    # Generate embeddings
    embeddings = client.generate_embeddings(
        windows,           # (n_windows, window_len, n_features)
        batch_size=32,
        embedding_dim=128
    )
    # Returns: (n_windows, 128)
```

### Command Line

```bash
# Generate embeddings for subject 1
python generate_embeddings.py 1

# With custom settings
python generate_embeddings.py 1 --batch-size 64 --embedding-dim 256

# Using mock (no API calls)
python generate_embeddings.py 1 --mock
```

## Testing Without API Key

```python
from src.embeddings.api_client import MockAPIClient

# No API key needed
client = MockAPIClient(embedding_dim=128)
embeddings = client.generate_embeddings(windows)
```

## Common Tasks

### Check API Health
```python
status = client.check_health()
# {'status': 'healthy', 'version': '1.0.0'}
```

### Single Window
```python
embedding = client.generate_single_embedding(window)
# Shape: (embedding_dim,)
```

### Get Model Info
```python
info = client.get_embedding_info()
# Shows available models and dimensions
```

## Error Handling

```python
from src.embeddings.api_client import AuthenticationError, RateLimitError

try:
    embeddings = client.generate_embeddings(windows)
except AuthenticationError:
    print("Check your API key in .env")
except RateLimitError:
    print("Wait and retry - client auto-retries")
```

## Configuration Options

```python
client = APIClient(
    api_key="custom_key",           # Or reads from .env
    base_url="https://api.com/v1",  # Custom endpoint
    timeout=60,                      # Request timeout (seconds)
    max_retries=5,                   # Retry attempts
    retry_delay=2.0                  # Initial delay (seconds)
)
```

## Performance Tips

| Dataset Size | Recommended Batch Size |
|--------------|------------------------|
| < 100 windows | 64 |
| 100-1000 windows | 32 |
| > 1000 windows | 16-32 |

## File Locations

- **API Client:** `src/embeddings/api_client.py`
- **Tests:** `tests/test_api_client.py`
- **CLI Tool:** `generate_embeddings.py`
- **Config:** `.env` (create from `.env.example`)
- **Full Guide:** `docs/API_CLIENT_GUIDE.md`

## Troubleshooting

| Error | Solution |
|-------|----------|
| "API key not found" | Set `WOOD_WIDE_API_KEY` in `.env` |
| "Rate limit exceeded" | Client auto-retries; reduce batch size if persistent |
| "Request timeout" | Increase `timeout` parameter |
| "Connection error" | Check internet connection and API status |

## Quick Test

```bash
# Run all tests (should pass 18/18)
pytest tests/test_api_client.py -v

# Test with mock client
python generate_embeddings.py 1 --mock
```

## Data Flow

```
Preprocessed Windows (114, 960, 5)
         ↓
    APIClient.generate_embeddings()
         ↓
   Batch 1: (32, 960, 5) → API → (32, 128)
   Batch 2: (32, 960, 5) → API → (32, 128)
   Batch 3: (32, 960, 5) → API → (32, 128)
   Batch 4: (18, 960, 5) → API → (18, 128)
         ↓
    Concatenate
         ↓
    Embeddings (114, 128)
```

## Next Steps

After generating embeddings:
1. Analyze for signal decoupling
2. Detect anomalies
3. Visualize in dashboard
