# CLAUDE.md

## What this project is

"Health Sync Monitor" is a technical demo for Wood Wide AI. Shows why simple HR thresholds don't work for health monitoring (too many false alarms during exercise) and how multivariate embeddings fix that.

The idea is embedding windows of HR + activity data via the Wood Wide API instead of `if hr > 100: alert()`. The embedding captures context, so high HR during cycling looks different from high HR during sleep.

Built on the PPG-DaLiA dataset (real wearable data from 15 subjects).

## How it's structured

```
src/
  ingestion/     — preprocess raw PPG-DaLiA .pkl files into windowed arrays
  embeddings/    — Wood Wide API client + embedding generation
  detectors/     — anomaly detectors (threshold, isolation forest, woodwide centroid)
app.py           — Streamlit dashboard (main entry point)
app_content.py   — display text for the dashboard
app_code_snippets.py — code snippet rendering utilities
tests/           — pytest suite (run with `pytest`)
```

## Running it

```bash
# setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# dashboard
streamlit run app.py

# tests
pytest
```

Use `python3` on this machine, not `python`.

## Working with the Wood Wide API

API client lives in `src/embeddings/api_client.py`. The flow is: upload CSV → train model → poll until done → run inference.

- API key goes in `.env` as `WOOD_WIDE_API_KEY`. Never hardcode it.
- Base URL is `https://beta.woodwide.ai` (not `api.woodwide.ai`).
- For local dev/testing, use `MockAPIClient` or pass `--mock` to CLI scripts.
- The API chokes on >500 CSV columns, so `_extract_summary_features()` computes 36 stats per window before uploading.
- `InsufficientCreditsError` is raised on HTTP 402 — the Streamlit app handles this with a dialog prompting for a new key.

## Code style

- Keep modules separate: ingestion, embeddings, detection, and visualization don't import each other sideways.
- Write clear function docstrings. Type hints where they help.
- Don't hardcode paths or API keys.
- Prefer editing existing files over creating new ones.
- When changing the Streamlit app, keep display text in `app_content.py` and layout logic in `app.py`.

## Data pipeline

```bash
# get data
python3 download_dataset.py           # real PPG-DaLiA data
python3 create_realistic_synthetic_data.py  # or synthetic for quick testing

# preprocess
python3 process_all_subjects.py

# generate embeddings
python3 generate_embeddings.py 1 --mock      # mock API
python3 generate_embeddings.py 1 --batch-size 32  # real API

# run detection
python3 woodwide_detection.py 1 --use-mock --compare-baseline 100
```
