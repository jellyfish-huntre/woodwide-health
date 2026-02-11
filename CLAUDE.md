# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Health Sync Monitor is a developer-facing technical demo for Woodwide AI that demonstrates how multivariate embeddings solve the "context problem" in wearable health monitoring. Unlike traditional fitness trackers that use simple heart rate thresholds (causing false alarms during exercise), this application uses Wood Wide AI to understand the relationship between heart rate and physical activity, only alerting when signals become "decoupled" (e.g., racing heart during sleep).

## Core Architecture

The application follows a modular **"data-to-dashboard" pipeline**:

1. **Data Ingestion** → High-frequency time-series data (heart rate, activity metrics)
2. **Embedding Generation** → Transform data into latent representations via Wood Wide API
3. **Signal Analysis** → Detect decoupling between heart rate and activity context
4. **Visualization** → Display health states and alerts

### Key Architectural Concept

The core innovation is using **multivariate embeddings** to understand contextual relationships:

- Traditional: `if heart_rate > threshold: alert()`
- This app: Embeddings learn that high HR during exercise is normal, but high HR during sleep indicates decoupling

## Technology Stack

- **Python**: Core data processing and API integration
- **Pandas**: Time-series data ingestion and manipulation
- **NumPy**: Numerical operations on health metrics
- **Wood Wide API**: Critical component for embedding generation
- **Streamlit**: Data dashboard and visualization
- **React**: Additional UI components/interface

## Wood Wide API Integration

The Wood Wide API is the key component that transforms raw health metrics into latent representations.

### APIClient Class

Located in `src/embeddings/api_client.py`, this class handles all API interactions:

- **Authentication:** Reads `WOOD_WIDE_API_KEY` from `.env` file
- **Batching:** Automatically batches large requests (default: 32 windows per batch)
- **Retry Logic:** Exponential backoff with configurable max retries
- **Error Handling:** Custom exceptions for auth, rate limiting, and network errors
- **MockAPIClient:** Use for testing without real API calls (`--mock` flag)

### API Data Flow

- **Input:** Preprocessed windows of shape `(n_windows, window_length, n_features)`
- **Output:** Embeddings of shape `(n_windows, embedding_dim)`
- **Normalization:** Embeddings are unit-normalized (L2 norm = 1)
- **Rate Limiting:** Client respects `Retry-After` headers automatically

See `docs/API_CLIENT_GUIDE.md` for detailed usage examples.

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run interactive Streamlit dashboard
streamlit run app.py

# The dashboard includes:
# - Dataset overview and activity timeline
# - Baseline threshold detection (shows the problem)
# - Wood Wide embedding detection (shows the solution)
# - Side-by-side comparison (94% improvement in false positive rate)

# Run with specific port
streamlit run app.py --server.port 8501

# See docs/STREAMLIT_APP_GUIDE.md for detailed usage
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Data Pipeline Development

```bash
# Download dataset (or create synthetic data for testing)
python download_dataset.py
# OR for quick testing with realistic HR patterns:
python create_realistic_synthetic_data.py

# Preprocess single subject
python -m src.ingestion.preprocess

# Preprocess all subjects
python process_all_subjects.py

# Verify data quality
python verify_data_quality.py 1

# Run baseline threshold detection (demonstrates the problem)
python baseline_threshold_detection.py 1 --threshold 100 --save-plot plot.png

# Generate embeddings (using mock API for testing)
python generate_embeddings.py 1 --mock

# Generate embeddings (using real Wood Wide API)
python generate_embeddings.py 1 --batch-size 32

# Run Wood Wide detector (embedding-based, solves the problem)
python woodwide_detection.py 1 --use-mock --compare-baseline 100 --save-plot plot.png

# Visualize preprocessed data
python visualize_data.py 1 --window 10
```

## Project Structure Notes

- Keep data ingestion, embedding generation, and visualization as separate, composable modules
- write detailed, clear function specs and keep the codebase easy to understand, ready for change, and safe from bugs
- do not hardcode API keys; use a .env file
- use uv or pip for dependency management
- The pipeline should handle high-frequency data efficiently (expect 1Hz+ sampling rates)
- API calls to Wood Wide should be batched appropriately to handle time-series data
- Alert logic lives in the analysis module, separate from embedding generation
