# API Response Handling Tests

## Overview

Comprehensive test suite for verifying that the APIClient correctly handles successful JSON responses from the Wood Wide API endpoint. Uses mocking to test without real API calls.

## Test Results

✅ **15/15 tests passed** (100% success rate)

## Test Coverage

### 1. Successful JSON Response Tests (9 tests)

#### `test_successful_embedding_response`
**Purpose:** Verify complete embedding generation workflow

**What it tests:**
- Request is made with correct HTTP method (POST)
- Request URL includes `/embeddings` endpoint
- Request payload contains windows and config
- Response JSON is correctly parsed
- Embeddings are extracted as numpy array
- Output shape matches expected dimensions (5, 128)

**Mock setup:**
```python
mock_response = Mock()
mock_response.status_code = 200
mock_response.json.return_value = {
    "embeddings": [[...], [...], ...],  # 5x128 embeddings
    "model": "multivariate-v1",
    "status": "success"
}
```

**Verifies:**
- ✅ Request method: POST
- ✅ Endpoint: `/embeddings`
- ✅ Payload structure
- ✅ Response parsing
- ✅ Array conversion
- ✅ Shape validation

---

#### `test_health_check_response`
**Purpose:** Verify health check endpoint handling

**What it tests:**
- Health endpoint returns correct status
- JSON fields are accessible
- Response contains expected metadata

**Mock response:**
```python
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 12345
}
```

**Verifies:**
- ✅ Status field: "healthy"
- ✅ Version field: "1.0.0"
- ✅ Additional metadata: uptime

---

#### `test_embedding_info_response`
**Purpose:** Verify model information endpoint

**What it tests:**
- Info endpoint returns model list
- Multiple models can be returned
- Model metadata is accessible

**Mock response:**
```python
{
    "models": [
        {
            "name": "multivariate-v1",
            "embedding_dim": 128,
            "max_sequence_length": 1024
        },
        {
            "name": "multivariate-v2",
            "embedding_dim": 256,
            "max_sequence_length": 2048
        }
    ]
}
```

**Verifies:**
- ✅ Multiple models returned
- ✅ Model names
- ✅ Embedding dimensions
- ✅ Sequence length limits

---

#### `test_multiple_batches_successful_responses`
**Purpose:** Verify batching works correctly

**What it tests:**
- Client makes multiple requests for large datasets
- Batch sizes are correct (10, 10, 5 for 25 windows)
- Final concatenation works
- Total output shape is correct

**Scenario:**
- Input: 25 windows
- Batch size: 10
- Expected: 3 batches (10 + 10 + 5)

**Verifies:**
- ✅ Correct number of batches: 3
- ✅ Batch sizes: [10, 10, 5]
- ✅ Final shape: (25, 128)

---

#### `test_json_parsing_with_nested_data`
**Purpose:** Verify complex nested JSON is parsed

**What it tests:**
- Nested dictionaries are accessible
- Deep nesting doesn't break parsing
- Metadata can be extracted

**Mock response:**
```python
{
    "embeddings": [...],
    "metadata": {
        "model_version": "1.2.3",
        "processing_info": {
            "normalization": "L2",
            "timestamp": "2024-01-01T12:00:00Z"
        },
        "performance": {
            "inference_time_ms": 42.5
        }
    }
}
```

**Verifies:**
- ✅ Top-level fields accessible
- ✅ Nested fields accessible
- ✅ Deep nesting works
- ✅ Various data types (string, float)

---

#### `test_response_headers_accessible`
**Purpose:** Verify HTTP headers are available

**What it tests:**
- Response headers can be accessed
- Standard headers present
- Custom headers available

**Mock headers:**
```python
{
    'Content-Type': 'application/json',
    'X-Request-ID': 'req-123456',
    'X-RateLimit-Remaining': '100'
}
```

**Verifies:**
- ✅ Content-Type header
- ✅ Request ID tracking
- ✅ Rate limit info

---

#### `test_empty_embeddings_list_handled`
**Purpose:** Verify empty response handling

**What it tests:**
- Empty windows input handled
- Empty embeddings list returned
- Correct shape maintained (0, dim)

**Verifies:**
- ✅ Empty array shape: (0, 128)
- ✅ 2D array maintained
- ✅ No errors on empty input

---

#### `test_response_with_additional_metadata`
**Purpose:** Verify extra fields don't break parsing

**What it tests:**
- Unknown fields in response ignored
- Core functionality unaffected
- Embeddings still extracted correctly

**Mock response:**
```python
{
    "embeddings": [...],
    "extra_field_1": "ignored",
    "extra_field_2": {"nested": "also ignored"}
}
```

**Verifies:**
- ✅ Extra fields don't cause errors
- ✅ Embeddings still extracted
- ✅ Correct shape returned

---

#### `test_single_embedding_response`
**Purpose:** Verify single window handling

**What it tests:**
- Single embedding correctly handled
- Shape is 1D (128,) not 2D (1, 128)
- Convenience method works

**Verifies:**
- ✅ Single embedding shape: (128,)
- ✅ Numpy array type
- ✅ Method: `generate_single_embedding()`

---

### 2. Request Payload Validation (4 tests)

#### `test_request_payload_structure`
**Purpose:** Verify request format is correct

**What it tests:**
- Payload has required fields
- Windows are serialized to lists
- Config is included
- Embedding dimension is set

**Captured payload:**
```python
{
    "windows": [[...], [...], ...],  # List of lists
    "config": {
        "embedding_dim": 128,
        "normalize": True
    }
}
```

**Verifies:**
- ✅ `windows` field present
- ✅ `config` field present
- ✅ Windows converted to lists (JSON serializable)
- ✅ Config parameters set

---

#### `test_authorization_header_sent`
**Purpose:** Verify authentication header

**What it tests:**
- Authorization header is set
- Bearer token format used
- API key is included

**Verifies:**
- ✅ Header name: "Authorization"
- ✅ Format: "Bearer {api_key}"
- ✅ API key present

---

#### `test_content_type_header_sent`
**Purpose:** Verify Content-Type header

**What it tests:**
- Content-Type is application/json
- Header is set on session

**Verifies:**
- ✅ Content-Type: "application/json"

---

#### `test_user_agent_header_sent`
**Purpose:** Verify User-Agent header

**What it tests:**
- User-Agent identifies client
- Application name included

**Verifies:**
- ✅ User-Agent contains "HealthSyncMonitor"

---

### 3. Response Integrity Tests (2 tests)

#### `test_embedding_values_preserved`
**Purpose:** Verify data integrity during parsing

**What it tests:**
- Exact values are preserved
- No data loss in conversion
- Float values accurate

**Test data:**
```python
expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
```

**Verifies:**
- ✅ Values match exactly (5 decimal places)
- ✅ No rounding errors
- ✅ Array conversion preserves data

---

#### `test_float_precision_maintained`
**Purpose:** Verify float precision

**What it tests:**
- High-precision floats handled
- Precision within float32 limits
- Scientific notation works

**Test values:**
```python
[0.123456789, -0.987654321, 0.555555555]
```

**Verifies:**
- ✅ Precision to 6 decimal places
- ✅ Negative values handled
- ✅ Various magnitudes work

---

## Test Categories Summary

| Category | Tests | Purpose |
|----------|-------|---------|
| **Successful Responses** | 9 | Verify JSON parsing and data extraction |
| **Request Validation** | 4 | Verify request format and headers |
| **Data Integrity** | 2 | Verify data preservation and precision |
| **Total** | **15** | **Complete API response handling** |

## Mocking Strategy

### What is Mocked

1. **HTTP Requests**: `api_client.session.request`
2. **HTTP Responses**: `Mock()` objects with:
   - `status_code`: 200
   - `json()`: Returns test data
   - `headers`: Optional headers

### Why Mock?

- ✅ No real API calls needed
- ✅ Fast test execution
- ✅ Deterministic results
- ✅ Test edge cases easily
- ✅ No API key required
- ✅ No network dependency

### Mock Example

```python
with patch.object(api_client.session, 'request') as mock_request:
    # Create mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2], [0.3, 0.4]]
    }
    mock_request.return_value = mock_response

    # Call API client method
    embeddings = api_client.generate_embeddings(windows)

    # Verify behavior
    assert mock_request.called
    assert embeddings.shape == (2, 2)
```

## Coverage Report

### Response Scenarios Tested

- ✅ Single embedding
- ✅ Multiple embeddings (batch)
- ✅ Empty embeddings
- ✅ Multiple batches
- ✅ Nested JSON
- ✅ Additional metadata
- ✅ Health check
- ✅ Model info

### Request Scenarios Tested

- ✅ Authorization header
- ✅ Content-Type header
- ✅ User-Agent header
- ✅ Payload structure
- ✅ Windows serialization
- ✅ Config parameters

### Data Integrity Scenarios

- ✅ Value preservation
- ✅ Float precision
- ✅ Array conversion
- ✅ Shape validation

## Running the Tests

### Run all response handling tests
```bash
pytest tests/test_api_response_handling.py -v
```

### Run specific test class
```bash
pytest tests/test_api_response_handling.py::TestSuccessfulJSONResponse -v
```

### Run single test
```bash
pytest tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_successful_embedding_response -v
```

### Run with coverage
```bash
pytest tests/test_api_response_handling.py --cov=src.embeddings.api_client --cov-report=html
```

## Test Output

```
============================= test session starts ==============================
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_successful_embedding_response PASSED [  6%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_health_check_response PASSED [ 13%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_embedding_info_response PASSED [ 20%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_multiple_batches_successful_responses PASSED [ 26%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_json_parsing_with_nested_data PASSED [ 33%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_response_headers_accessible PASSED [ 40%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_empty_embeddings_list_handled PASSED [ 46%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_response_with_additional_metadata PASSED [ 53%]
tests/test_api_response_handling.py::TestSuccessfulJSONResponse::test_single_embedding_response PASSED [ 60%]
tests/test_api_response_handling.py::TestRequestPayloadValidation::test_request_payload_structure PASSED [ 66%]
tests/test_api_response_handling.py::TestRequestPayloadValidation::test_authorization_header_sent PASSED [ 73%]
tests/test_api_response_handling.py::TestRequestPayloadValidation::test_content_type_header_sent PASSED [ 80%]
tests/test_api_response_handling.py::TestRequestPayloadValidation::test_user_agent_header_sent PASSED [ 86%]
tests/test_api_response_handling.py::TestResponseIntegrity::test_embedding_values_preserved PASSED [ 93%]
tests/test_api_response_handling.py::TestResponseIntegrity::test_float_precision_maintained PASSED [100%]

============================== 15 passed in 1.27s ==============================
```

## Key Insights

### What These Tests Prove

1. **Correct Parsing**: Client correctly extracts embeddings from JSON
2. **Data Integrity**: Values preserved during JSON → numpy conversion
3. **Batching Works**: Multiple requests handled correctly
4. **Headers Set**: Authentication and content-type headers present
5. **Edge Cases**: Empty arrays, nested data, extra fields handled
6. **Precision**: Float values maintained with sufficient precision

### What's NOT Tested Here

These tests focus on **successful** responses. For error cases, see:
- `tests/test_api_client.py` - Error handling, retries, rate limits
- `tests/test_generate.py` - Validation and workflow tests

## Related Documentation

- **API Client Guide**: `docs/API_CLIENT_GUIDE.md`
- **Embedding Generation**: `docs/EMBEDDING_GENERATION.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
