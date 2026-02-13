# Health Sync Monitor

A developer-facing technical demo showing how Wood Wide AI's multivariate embeddings solve the "context problem" in wearable health monitoring.

Traditional fitness trackers alert on raw heart rate thresholds, producing 25-80% false positive rates during exercise. Health Sync Monitor uses Wood Wide embeddings to understand the relationship between heart rate and physical activity, achieving **5.1% exercise false positive rate** while maintaining comparable anomaly detection sensitivity.

See [WRITEUP.md](WRITEUP.md) for detailed problem framing, results, and analysis.

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API
cp .env.example .env   # then set WOOD_WIDE_API_KEY

# Prepare data
python3 download_dataset.py       # PPG-DaLiA from UCI
python3 process_all_subjects.py   # Preprocess into windows

# Generate embeddings
python3 generate_embeddings.py 1          # Real API
python3 generate_embeddings.py 1 --mock   # Mock (no API key needed)

# Launch dashboard
streamlit run app.py
```

## Tests

```bash
pytest                              # 81 tests
pytest --cov=src --cov-report=html  # With coverage
```

## Architecture

```
PPG-DaLiA Data  -->  Preprocessing  -->  Wood Wide API  -->  Centroid Detector  -->  Dashboard
(PPG + ACC)         (sync, window)      (embeddings)       (anomaly detection)    (Streamlit)
```

## Dataset

PPG-DaLiA: Reiss, A. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5N312
