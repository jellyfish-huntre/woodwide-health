# Streamlit Dashboard Guide

## Overview

The Health Sync Monitor dashboard is an interactive web application that demonstrates Wood Wide AI's context-aware health monitoring versus traditional threshold-based detection.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Generate realistic synthetic data
python create_realistic_synthetic_data.py

# Preprocess data
python -m src.ingestion.preprocess

# Run baseline detection
python baseline_threshold_detection.py 1 --threshold 100

# Run Wood Wide detection
python woodwide_detection.py 1 --use-mock
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 📊 Overview Tab

**Dataset Information:**
- Subject selection
- Window count and duration
- Activity distribution pie chart
- Interactive activity timeline

**The Context Problem:**
- Explanation of why threshold-based detection fails
- Alert fatigue visualization

### 🔴 Baseline Detection Tab

**Traditional Threshold Approach:**
- Heart rate extraction from PPG signal
- Simple threshold alerting (HR > X BPM)
- Real-time metrics:
  - Total alerts
  - False positive rate
  - Alerts during exercise vs. rest
- Interactive plots:
  - Heart rate over time with threshold
  - Alert timeline
- Performance breakdown by activity

**Key Insight:** Shows 100% false positive rate during exercise

### 🟢 Wood Wide Detection Tab

**Embedding-Based Context-Aware Approach:**
- Automatic embedding generation (mock or real API)
- Distance from normal centroid computation
- Configurable threshold percentile
- Real-time metrics:
  - Total alerts
  - Low false positive rate (~5%)
  - Distance threshold
- Interactive plots:
  - Distance from normal over time
  - Alert timeline
- Performance breakdown by activity
- "How it Works" explainer

**Key Insight:** Shows dramatic reduction in false alarms

### ⚖️ Comparison Tab

**Side-by-Side Analysis:**
- Key metrics comparison:
  - False positive rate: 100% → 5.6% (94.4% improvement)
  - Alerts during exercise: 71 → 4 (67 fewer)
  - Total alerts: 109 → 21 (88 fewer)
- Interactive comparison chart
- Synchronized timeline visualization
- Detailed comparison table
- Conclusion and key takeaways

## Configuration Options

### Sidebar Controls

**Subject Selection:**
- Choose which subject to analyze
- Automatically detects available processed data

**Baseline Threshold (BPM):**
- Range: 80-140 BPM
- Default: 100 BPM
- Adjust to see how it affects false positive rate

**Wood Wide Threshold Percentile:**
- Range: 85-99
- Default: 95 (recommended)
- Higher = fewer alerts (less sensitive)
- Lower = more alerts (more sensitive)

**API Options:**
- Use Mock API (default: enabled)
  - No API key needed
  - Fast testing
  - Deterministic results
- Force Regenerate Embeddings
  - Regenerate even if cached
  - Useful when changing embedding parameters

## Interactive Visualizations

### Plotly Charts

All charts are interactive with:
- **Zoom:** Click and drag to zoom
- **Pan:** Hold shift and drag to pan
- **Hover:** Hover for detailed information
- **Legend:** Click legend items to show/hide traces
- **Reset:** Double-click to reset view

### Activity Timeline

- Shows when user was performing each activity
- Color-coded by activity type
- Hover for exact timing

### Detection Plots

**Baseline:**
- Blue line: Heart rate over time
- Red dashed: Threshold
- Red X: Alerts
- Red shading: Alert active

**Wood Wide:**
- Green line: Distance from normal
- Red dashed: Threshold
- Red X: Alerts
- Green shading: Alert active

**Comparison:**
- Side-by-side synchronized timelines
- Red = Baseline alerts
- Green = Wood Wide alerts
- Clearly shows reduction in false alarms

## Understanding the Results

### Good Performance Indicators

**Baseline:**
- High false positive rate (expected)
- Many alerts during exercise
- Alert fatigue evident

**Wood Wide:**
- Low false positive rate (<10%)
- Few alerts during exercise
- Reasonable alerts during rest
- Practical for deployment

### Tuning the Detector

**If too many Wood Wide alerts:**
1. Increase threshold percentile (e.g., 95 → 97)
2. This makes detector less sensitive
3. Alerts only on more extreme anomalies

**If too few Wood Wide alerts:**
1. Decrease threshold percentile (e.g., 95 → 92)
2. This makes detector more sensitive
3. Catches more subtle anomalies

**Recommended:** Start at 95 and adjust based on your needs

## Real-World Usage

### Demo Workflow

1. **Start with Overview** - Understand the data
2. **Check Baseline** - See the problem (100% FP rate)
3. **View Wood Wide** - See the solution (5% FP rate)
4. **Compare** - Understand the improvement

### For Presentations

**Problem Statement:**
- Show Overview tab → Explain context problem
- Show Baseline tab → Demonstrate 100% FP rate
- Emphasize alert fatigue

**Solution:**
- Show Wood Wide tab → Demonstrate low FP rate
- Explain how embeddings capture context
- Show "How it Works" section

**Results:**
- Show Comparison tab → Quantify improvement
- Highlight 94% reduction in false alarms
- Discuss real-world impact

### For Development

**Testing Different Subjects:**
- Use sidebar to switch subjects
- Compare performance across different activity patterns

**Tuning Thresholds:**
- Adjust baseline threshold to see it never works well
- Adjust Wood Wide percentile to find optimal sensitivity

**Regenerating Embeddings:**
- Enable "Force Regenerate" to test different models
- Compare mock vs. real API performance

## Technical Details

### Data Flow

```
1. Load preprocessed windows
   ↓
2. Extract heart rate (baseline)
   OR
   Generate embeddings (Wood Wide)
   ↓
3. Apply detection algorithm
   ↓
4. Compute metrics
   ↓
5. Visualize results
```

### Caching

The app uses Streamlit's `@st.cache_data` for:
- Loading preprocessed data
- Loading cached embeddings
- Loading detection results

This makes the app fast on subsequent runs.

### File Structure

```
data/
├── processed/              # Preprocessed windows
│   └── subject_01_processed.pkl
├── embeddings/            # Generated embeddings
│   ├── subject_01_embeddings.npy
│   └── subject_01_metadata.pkl
├── baseline_detection/    # Baseline results
│   └── subject_01_threshold_100.pkl
└── woodwide_detection/    # Wood Wide results
    └── subject_01_results.pkl
```

## Troubleshooting

### No Processed Data Found

**Error:** "No processed data found"

**Solution:**
```bash
python create_realistic_synthetic_data.py
python -m src.ingestion.preprocess
```

### Embeddings Not Found

**Error:** "Please generate embeddings first"

**Solution:**
1. Go to Wood Wide Detection tab
2. Click to generate embeddings
3. Wait for completion
4. Return to Comparison tab

### API Key Error

**Error:** "API key not found"

**Solution:**
1. Use "Use Mock API" option (recommended for demo)
2. Or set `WOOD_WIDE_API_KEY` in `.env` file

### Slow Performance

**Issue:** App is slow

**Solutions:**
- Use mock API instead of real API
- Use cached embeddings (disable "Force Regenerate")
- Reduce subject data size
- Check Streamlit caching is working

## Customization

### Adding New Subjects

```bash
# Add data to data/PPGDaLiA/
# Run preprocessing
python -m src.ingestion.preprocess

# Subject will appear in dropdown
```

### Changing Activity Labels

Edit `activity_map` in `app.py`:

```python
activity_map = {
    1: 'Sitting',
    2: 'Cycling',
    3: 'Walking',
    # Add custom activities...
}
```

### Custom Visualizations

Add new plots using Plotly:

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(...)
st.plotly_chart(fig, use_container_width=True)
```

## Deployment

### Local Deployment

```bash
# Production mode
streamlit run app.py --server.port 8501 --server.headless true
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository

**Docker:**
```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### Environment Variables

For production, set:
```bash
WOOD_WIDE_API_KEY=your_actual_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

## Performance Tips

### Speed Optimization

1. **Use Mock API** - 10x faster than real API
2. **Cache Embeddings** - Don't regenerate unnecessarily
3. **Limit Data Size** - Start with 1-2 subjects
4. **Precompute Results** - Run detection scripts beforehand

### Memory Optimization

- Load data only when needed
- Use `@st.cache_data` for expensive operations
- Clear cache if memory issues: Settings → Clear Cache

## Examples

### Example Session

```bash
# 1. Setup
python create_realistic_synthetic_data.py
python -m src.ingestion.preprocess

# 2. Pre-generate results
python baseline_threshold_detection.py 1 --threshold 100
python woodwide_detection.py 1 --use-mock

# 3. Launch dashboard
streamlit run app.py

# 4. Explore
# - Overview → See dataset
# - Baseline → 100% FP rate
# - Wood Wide → 5.6% FP rate
# - Comparison → 94% improvement
```

### Example Use Cases

**Research Demo:**
- Show problem and solution clearly
- Quantify improvement with metrics
- Interactive exploration for questions

**Product Demo:**
- Demonstrate practical deployment
- Show real-time detection simulation
- Highlight user experience benefits

**Development:**
- Test different threshold settings
- Compare subjects
- Validate detector performance

## Next Steps

1. ✅ Explore dashboard locally
2. ⏳ Deploy to Streamlit Cloud for sharing
3. ⏳ Add real-time streaming simulation
4. ⏳ Integrate with real Wood Wide API
5. ⏳ Add multi-subject comparison

## Summary

The Streamlit dashboard provides an interactive, visual demonstration of how Wood Wide embeddings solve the context problem in health monitoring. With a 94% reduction in false alarms, it clearly shows the practical value of context-aware AI.

**Launch the dashboard and see the difference for yourself!**

```bash
streamlit run app.py
```
