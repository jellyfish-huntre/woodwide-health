# Health Sync Monitor

A developer-facing technical demo for Woodwide AI showing how multivariate embeddings solve the "context problem" in wearable health monitoring.

## Overview

Traditional fitness trackers use simple heart rate thresholds that trigger false alarms during exercise. Health Sync Monitor uses Wood Wide AI to understand the relationship between heart rate and physical activity, only alerting when signals become "decoupled" (e.g., racing heart during sleep).

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download PPG-DaLiA dataset from UCI repository
python download_dataset.py
```

This downloads the PPG-DaLiA dataset containing:
- 15 subjects performing 8 different activities
- PPG (heart rate) at 64 Hz
- Accelerometer (movement) at 32 Hz
- Ground truth activity labels

### 3. Configure API

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Wood Wide API key
# WOOD_WIDE_API_KEY=your_api_key_here
```

Get your API key from [Wood Wide AI Dashboard](https://woodwide.ai/dashboard).

### 4. Preprocess Data

```bash
# Process a single subject
python -m src.ingestion.preprocess

# Or process all subjects
python process_all_subjects.py
```

This creates synchronized, windowed feature vectors ready for embedding generation.

### 5. Generate Embeddings

```bash
# Using real Wood Wide API
python generate_embeddings.py 1

# Using mock API (for testing without API key)
python generate_embeddings.py 1 --mock
```

This transforms time-series windows into multivariate embeddings.

### 6. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Architecture

```
Data Ingestion → Embedding Generation → Signal Analysis → Visualization
     ↓                    ↓                    ↓               ↓
  PPG-DaLiA        Wood Wide API      Decoupling Detector   Streamlit
  (32 Hz)         (Embeddings)         (Anomaly)          (Dashboard)
```

## Key Concepts

### Multivariate Embeddings

Instead of using simple thresholds:
```python
# Traditional approach (many false alarms)
if heart_rate > 120:
    alert("High heart rate!")
```

We use embeddings to understand context:
```python
# Embedding-based approach (context-aware)
embedding = wood_wide_api.embed([heart_rate, activity_level])
if is_decoupled(embedding):
    alert("Unusual heart rate for current activity")
```

### Signal Synchronization

The preprocessor handles different sampling rates:
- **PPG**: 64 Hz → downsampled to 32 Hz
- **Accelerometer**: 32 Hz → native rate
- **Synchronized**: All signals aligned at 32 Hz

### Rolling Windows

Creates fixed-length feature vectors:
- **Window size**: 30 seconds (default)
- **Stride**: 5 seconds (overlapping windows)
- **Output**: `(n_windows, window_length, n_features)` arrays

## Project Structure

```
health/
├── src/
│   ├── ingestion/          # Data loading and preprocessing
│   ├── embeddings/         # Wood Wide API integration
│   ├── analysis/           # Decoupling detection
│   └── visualization/      # Streamlit dashboard
├── tests/                  # Test suite
├── data/
│   ├── raw/               # Downloaded PPG-DaLiA dataset
│   └── processed/         # Windowed feature vectors
├── download_dataset.py    # Dataset download script
└── requirements.txt       # Python dependencies
```

## Next Steps

1. ✅ Download and preprocess data
2. ⏳ Integrate Wood Wide API for embedding generation
3. ⏳ Implement decoupling detection algorithm
4. ⏳ Build Streamlit visualization dashboard
5. ⏳ Add real-time data streaming capability

## Dataset Citation

PPG-DaLiA Dataset:
Attila Reiss. (2019). PPG-DaLiA. UCI Machine Learning Repository. https://doi.org/10.24432/C5N312
