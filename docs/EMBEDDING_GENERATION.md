# Embedding Generation Guide

## Overview

This guide covers the `send_windows_to_woodwide()` function and the complete embedding generation workflow for transforming preprocessed time-series health data into multivariate embeddings.

## Quick Start

```python
from src.embeddings.generate import send_windows_to_woodwide

# Send windowed data to Wood Wide API
embeddings, metadata = send_windows_to_woodwide(
    windows,              # Shape: (n_windows, window_length, n_features)
    batch_size=32,
    embedding_dim=128,
    use_mock=True        # Use mock for testing
)
```

## Function Reference

### `send_windows_to_woodwide()`

Main function for sending windowed data to the Wood Wide API endpoint.

**Signature:**
```python
send_windows_to_woodwide(
    windows: np.ndarray,
    api_key: Optional[str] = None,
    batch_size: int = 32,
    embedding_dim: Optional[int] = None,
    validate_input: bool = True,
    use_mock: bool = False
) -> Tuple[np.ndarray, Dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windows` | `np.ndarray` | Required | Input data, shape `(n_windows, window_length, n_features)` |
| `api_key` | `str` | `None` | API key (reads from env if not provided) |
| `batch_size` | `int` | `32` | Windows per API request |
| `embedding_dim` | `int` | `None` | Target dimension (API default if None) |
| `validate_input` | `bool` | `True` | Validate input data |
| `use_mock` | `bool` | `False` | Use mock API (no real calls) |

**Returns:**

Tuple of `(embeddings, metadata)`:
- `embeddings`: `np.ndarray` of shape `(n_windows, embedding_dim)`
- `metadata`: `dict` with generation statistics

**Raises:**
- `ValueError`: If input validation fails
- `WoodWideAPIError`: If API communication fails

## Complete Workflow

### Step 1: Prepare Data

Ensure you have preprocessed windowed data:

```python
import pickle

# Load preprocessed subject data
with open('data/processed/subject_01_processed.pkl', 'rb') as f:
    data = pickle.load(f)

windows = data['windows']  # Shape: (114, 960, 5)
```

### Step 2: Send to API

```python
from src.embeddings.generate import send_windows_to_woodwide

# Generate embeddings
embeddings, metadata = send_windows_to_woodwide(
    windows,
    batch_size=32,
    embedding_dim=128
)

print(f"Generated {len(embeddings)} embeddings")
# Output: Generated 114 embeddings
```

### Step 3: Inspect Results

```python
print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: (114, 128)

print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
# Output: Processing time: 0.52s

print(f"Throughput: {metadata['windows_per_second']:.1f} windows/sec")
# Output: Throughput: 220.7 windows/sec
```

## Input Validation

The function automatically validates input data:

### What Gets Validated

1. **Dimensionality**: Must be 3D array `(n_windows, window_length, n_features)`
2. **Null Values**: No NaN or infinite values
3. **Window Count**: At least 1 window
4. **Feature Count**: Warning if not 5 features (PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG)
5. **Value Ranges**: Warning for extreme values

### Example Validation

```python
# Valid data - passes
valid_windows = np.random.randn(10, 960, 5)
embeddings, _ = send_windows_to_woodwide(valid_windows, use_mock=True)
# ✓ Success

# Invalid data - fails
invalid_windows = np.random.randn(10, 960)  # 2D instead of 3D
try:
    embeddings, _ = send_windows_to_woodwide(invalid_windows, use_mock=True)
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Validation error: Windows must be 3D array
```

### Disable Validation

```python
# Skip validation (not recommended)
embeddings, _ = send_windows_to_woodwide(
    windows,
    validate_input=False,
    use_mock=True
)
```

## Batch Processing

### Choosing Batch Size

| Dataset Size | Recommended Batch Size | Rationale |
|--------------|------------------------|-----------|
| < 50 windows | 16 | Small overhead, fast |
| 50-200 windows | 32 | Balanced (default) |
| 200-1000 windows | 32-64 | Efficient batching |
| > 1000 windows | 16-32 | Memory management |

### Example

```python
# Small dataset
small_windows = np.random.randn(30, 960, 5)
embeddings, _ = send_windows_to_woodwide(
    small_windows,
    batch_size=16,
    use_mock=True
)

# Large dataset
large_windows = np.random.randn(500, 960, 5)
embeddings, _ = send_windows_to_woodwide(
    large_windows,
    batch_size=32,
    use_mock=True
)
```

## High-Level Functions

### Process Complete Subject

```python
from src.embeddings.generate import process_subject_windows

# Load → Validate → Generate → Save (all in one)
embeddings, metadata = process_subject_windows(
    subject_id=1,
    batch_size=32,
    use_mock=True
)

# Embeddings saved to:
#   data/embeddings/subject_01_embeddings.npy
#   data/embeddings/subject_01_metadata.pkl
```

### Batch Process Multiple Subjects

```python
from src.embeddings.generate import batch_process_subjects

# Process subjects 1, 2, 3
results = batch_process_subjects(
    subject_ids=[1, 2, 3],
    batch_size=32,
    use_mock=True
)

for subject_id, (embeddings, metadata) in results.items():
    print(f"Subject {subject_id}: {embeddings.shape}")
```

### Load Saved Embeddings

```python
from src.embeddings.generate import load_embeddings

# Load previously generated embeddings
embeddings, metadata = load_embeddings(subject_id=1)

print(f"Loaded: {embeddings.shape}")
print(f"Subject: {metadata['subject_id']}")
```

## Metadata Structure

The function returns detailed metadata:

```python
{
    'n_windows': 114,                    # Number of windows processed
    'window_shape': (960, 5),            # Shape of each window
    'embedding_dim': 128,                # Embedding dimension
    'batch_size': 32,                    # Batch size used
    'processing_time_seconds': 0.52,     # Total processing time
    'windows_per_second': 220.7,         # Throughput
    'api_status': 'healthy',             # API health status
    'mock': True                         # Whether mock was used
}
```

## Error Handling

### Common Errors

**1. Missing API Key**
```python
# Error: API key not found
# Solution: Set WOOD_WIDE_API_KEY in .env or pass api_key parameter
embeddings, _ = send_windows_to_woodwide(
    windows,
    api_key="your_key_here"
)
```

**2. Invalid Input Shape**
```python
# Error: Windows must be 3D array
# Solution: Ensure shape is (n_windows, window_length, n_features)
windows = windows.reshape(n_windows, window_length, n_features)
```

**3. NaN Values**
```python
# Error: Found X null/NaN values
# Solution: Clean data or investigate preprocessing
windows = np.nan_to_num(windows)  # Replace NaN with 0
```

**4. Rate Limiting**
```python
# Error: Rate limit exceeded
# Solution: Reduce batch_size or add delays
embeddings, _ = send_windows_to_woodwide(
    windows,
    batch_size=16  # Smaller batches
)
```

## Performance Optimization

### 1. Optimal Batch Size

```python
# Test different batch sizes
import time

for batch_size in [16, 32, 64]:
    start = time.time()
    embeddings, _ = send_windows_to_woodwide(
        windows,
        batch_size=batch_size,
        use_mock=True
    )
    elapsed = time.time() - start
    print(f"Batch size {batch_size}: {elapsed:.2f}s")
```

### 2. Disable Validation for Large Datasets

```python
# Skip validation if you've already validated
embeddings, _ = send_windows_to_woodwide(
    windows,
    validate_input=False  # Skip validation
)
```

### 3. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_subject(subject_id):
    return process_subject_windows(subject_id, use_mock=True)

# Process multiple subjects in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_subject, [1, 2, 3]))
```

## Testing Without API

### Mock API Client

```python
# Use mock=True to test without real API calls
embeddings, metadata = send_windows_to_woodwide(
    windows,
    use_mock=True  # No API key needed
)

# Mock generates deterministic random embeddings
# Same windows → same embeddings (for reproducibility)
```

### Example Test

```python
import numpy as np

# Create test data
test_windows = np.random.randn(10, 960, 5).astype(np.float32)

# Test with mock
embeddings, metadata = send_windows_to_woodwide(
    test_windows,
    batch_size=5,
    embedding_dim=64,
    use_mock=True
)

assert embeddings.shape == (10, 64)
assert metadata['mock'] is True
print("✓ Test passed!")
```

## Real-World Example

```python
"""
Complete real-world workflow: Subject 1 from preprocessing to embeddings.
"""

from src.embeddings.generate import process_subject_windows
import numpy as np

# Process subject 1
print("Generating embeddings for subject 1...")

embeddings, metadata = process_subject_windows(
    subject_id=1,
    data_dir="data/processed",
    output_dir="data/embeddings",
    batch_size=32,
    embedding_dim=128,
    validate_input=True,
    use_mock=False  # Use real API
)

# Analyze results
print(f"\n✓ Generated {len(embeddings)} embeddings")
print(f"  Dimension: {embeddings.shape[1]}")
print(f"  Processing time: {metadata['generation_metadata']['processing_time_seconds']:.2f}s")

# Compute embedding statistics
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nEmbedding statistics:")
print(f"  Mean norm: {norms.mean():.3f}")
print(f"  Std norm: {norms.std():.3f}")

# Check activity distribution
labels = metadata['labels']
unique_activities = np.unique(labels)
print(f"\nActivities: {unique_activities}")

for activity in unique_activities:
    count = (labels == activity).sum()
    print(f"  Activity {activity}: {count} windows")
```

## Next Steps

After generating embeddings:

1. **Analyze Signal Decoupling**: Use embeddings to detect when heart rate and activity become decoupled
2. **Visualize Embeddings**: Plot embeddings in 2D/3D using dimensionality reduction
3. **Train Classifier**: Use embeddings as features for activity classification
4. **Build Dashboard**: Display real-time embeddings in Streamlit

## See Also

- **API Client Guide**: `docs/API_CLIENT_GUIDE.md`
- **Quick Reference**: `API_QUICK_REFERENCE.md`
- **Example Scripts**: `examples/embedding_workflow.py`
- **Tests**: `tests/test_generate.py`
