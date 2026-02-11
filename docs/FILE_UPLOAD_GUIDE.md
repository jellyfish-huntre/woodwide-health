# File Upload Guide - Streamlit Dashboard

## Overview

The Streamlit dashboard now supports uploading your own PPG-DaLiA CSV files for real-time analysis. This allows you to test the Wood Wide detector on custom data without pre-processing.

## Quick Start

### 1. Launch the Dashboard

```bash
streamlit run app.py
```

### 2. Select Upload Mode

In the sidebar:
1. Choose **"Upload CSV"** under Data Source
2. Click **"📥 Download Sample CSV"** to see the expected format
3. Upload your own CSV file

### 3. Configure Preprocessing

Adjust parameters:
- **Window Length**: 10-60 seconds (default: 30s)
- **Stride**: 1-30 seconds (default: 5s)

### 4. Preprocess Data

Click **"🔄 Preprocess Data"** to convert raw signals into windows

### 5. Run Detection

Click **"🚀 Run Detection"** to analyze with both baseline and Wood Wide methods

## CSV Format Requirements

### Required Columns

Your CSV must include these columns (case-insensitive):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ppg` | float | PPG signal values | `0.123` |
| `accX` | float | Accelerometer X-axis | `-0.456` |
| `accY` | float | Accelerometer Y-axis | `0.789` |
| `accZ` | float | Accelerometer Z-axis | `9.812` |
| `label` | int | Activity label (1-7) | `2` |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_acc` | float | Timestamp in seconds |

### Activity Labels

Standard PPG-DaLiA activity codes:

| Label | Activity |
|-------|----------|
| 1 | Sitting |
| 2 | Cycling |
| 3 | Walking |
| 4 | Ascending stairs |
| 5 | Descending stairs |
| 6 | Vacuum cleaning |
| 7 | Ironing |

### Example CSV

```csv
timestamp_acc,ppg,accX,accY,accZ,label
0.0,0.123,-0.456,0.789,9.812,1
0.03125,0.145,-0.423,0.801,9.805,1
0.0625,0.167,-0.398,0.815,9.798,1
...
```

## Sample CSV

The dashboard provides a downloadable sample CSV that demonstrates the correct format:

1. Click **"📥 Download Sample CSV"** in the sidebar
2. Inspect the file to understand the structure
3. Use it as a template for your own data

**Sample includes:**
- 2 minutes of synthetic data
- PPG signal (simulated heartbeat)
- Accelerometer data (simulated movement)
- Activity labels (sitting → cycling)

## Data Processing Pipeline

### 1. Upload & Parse

```
CSV File → Parse columns → Validate format → Extract signals
```

**Checks:**
- Required columns present
- Data types correct
- Values are numeric

### 2. Synchronize Signals

```
Raw PPG (64 Hz) + ACC (32 Hz) → Resampled to common rate (32 Hz)
```

**Process:**
- Linear interpolation for synchronization
- Timestamp alignment
- Label propagation

### 3. Compute Features

```
Synchronized data → Add derived features → ACC magnitude
```

**Features added:**
- ACC_MAG = √(accX² + accY² + accZ²)

### 4. Create Windows

```
Continuous signals → Rolling windows → (n_windows, 960, 5)
```

**Parameters:**
- Window length: 30s × 32 Hz = 960 samples
- Stride: 5s × 32 Hz = 160 samples
- Overlap: ~83%

### 5. Ready for Detection

```
Windows → Baseline + Wood Wide detection → Results
```

## Preprocessing Parameters

### Window Length

**What it is:** Length of each analysis window in seconds

**Impact:**
- **Shorter (10-20s):** More granular analysis, more windows
- **Longer (40-60s):** Smoother analysis, fewer windows
- **Recommended: 30s** - Good balance

### Stride

**What it is:** Time between consecutive windows

**Impact:**
- **Shorter (1-3s):** High overlap, more windows, slower processing
- **Longer (10-30s):** Less overlap, fewer windows, faster processing
- **Recommended: 5s** - Good overlap for smooth detection

### Example Calculations

**2-minute recording with 30s windows and 5s stride:**

```
Duration: 120 seconds
Window length: 30 seconds
Stride: 5 seconds

Number of windows = (120 - 30) / 5 + 1 = 19 windows
```

## Workflow Examples

### Example 1: Quick Test with Sample Data

```
1. Select "Upload CSV"
2. Click "📥 Download Sample CSV"
3. Upload the downloaded file
4. Keep default parameters (30s window, 5s stride)
5. Click "🔄 Preprocess Data"
6. Click "🚀 Run Detection"
7. View results in all tabs
```

**Expected result:** 19 windows processed in ~1 second

### Example 2: Custom Data Analysis

```
1. Prepare your CSV with required columns
2. Select "Upload CSV"
3. Upload your file
4. Adjust parameters if needed:
   - Window: 30s
   - Stride: 5s
5. Click "🔄 Preprocess Data"
6. Review preprocessing results
7. Click "🚀 Run Detection"
8. Compare baseline vs Wood Wide
```

### Example 3: Parameter Tuning

```
1. Upload data
2. Try different window lengths:
   - 20s (more sensitive to short events)
   - 30s (balanced)
   - 40s (more stable, less noise)
3. Preprocess and detect for each
4. Compare results across parameters
```

## Troubleshooting

### Error: "Missing required columns"

**Problem:** CSV doesn't have all required columns

**Solution:**
1. Download sample CSV to see correct format
2. Ensure your CSV has: ppg, accX, accY, accZ, label
3. Check column names (case doesn't matter)

### Error: "Failed to parse CSV"

**Problem:** CSV format is invalid

**Solutions:**
- Ensure proper CSV formatting (commas, no extra quotes)
- Check for non-numeric values in signal columns
- Verify labels are integers (1-7)
- Remove any header rows except the column names

### Error: "Failed to preprocess data"

**Problem:** Data has incompatible sampling rates or lengths

**Solutions:**
- Ensure signals are same length or provide timestamps
- Check for NaN or infinite values
- Verify sampling rates are reasonable (e.g., 32-64 Hz)

### Warning: "Created 0 windows"

**Problem:** Data is too short for chosen window parameters

**Solution:**
- Reduce window length (e.g., 10s instead of 30s)
- Or increase stride
- Or collect more data

### Slow Preprocessing

**Problem:** Large files take time to process

**Solutions:**
- Start with smaller files to test
- Increase stride to create fewer windows
- Use shorter window lengths
- Consider pre-processing offline

## Performance Tips

### Optimal File Sizes

| Duration | File Size | Windows (30s/5s) | Processing Time |
|----------|-----------|------------------|-----------------|
| 2 min | ~10 KB | ~20 | ~1 sec |
| 5 min | ~25 KB | ~50 | ~2 sec |
| 10 min | ~50 KB | ~110 | ~3 sec |
| 30 min | ~150 KB | ~350 | ~8 sec |

### Recommendations

**For Testing:**
- Use 2-5 minute samples
- Default parameters (30s window, 5s stride)
- Mock API for embeddings

**For Analysis:**
- 10-30 minute recordings
- Adjust parameters based on activity type
- Real API if available

**For Production:**
- Pre-process offline
- Use cached results
- Deploy with optimized backend

## Advanced Features

### Session State

The app uses Streamlit session state to:
- Cache uploaded data across reruns
- Persist preprocessing results
- Remember user selections

**Benefits:**
- No need to re-upload after parameter changes
- Fast switching between tabs
- Smooth user experience

### Run Detection Button

**Purpose:** Trigger detection on-demand

**When to use:**
- After uploading new data
- After changing thresholds
- To regenerate results

**What it does:**
1. Extracts heart rate from PPG
2. Runs baseline threshold detection
3. Generates Wood Wide embeddings
4. Runs Wood Wide detector
5. Computes performance metrics
6. Updates all visualizations

## Integration with Pre-processed Data

You can switch between uploaded and pre-processed data:

1. **Pre-processed Data:**
   - Fast loading (already windowed)
   - No preprocessing step
   - Cached embeddings available

2. **Upload CSV:**
   - Process raw data
   - Custom parameters
   - Fresh embeddings

Both modes use the same detection algorithms and visualizations.

## Export Results

After detection, you can:

1. **Download Charts:**
   - Hover over any Plotly chart
   - Click camera icon to download PNG

2. **Screenshot Dashboard:**
   - Use browser screenshot tools
   - Capture specific tabs

3. **Copy Metrics:**
   - Select and copy text from results
   - Save to spreadsheet

## Security and Privacy

### Data Handling

- ✅ Files processed in-memory only
- ✅ No data stored on server
- ✅ No data transmitted externally (mock API mode)
- ✅ Session data cleared on browser close

### API Usage

- **Mock API (default):** No external calls, fully local
- **Real API:** Only if you enable and provide API key

### Best Practices

- Don't upload sensitive health data to public deployments
- Use mock API for demonstrations
- Deploy privately for real patient data
- Follow HIPAA/GDPR requirements for production

## Next Steps

### After Uploading Data

1. **Explore Results:**
   - Check all tabs (Overview, Baseline, Wood Wide, Comparison)
   - Hover over charts for details
   - Adjust thresholds and re-run

2. **Tune Parameters:**
   - Try different window lengths
   - Experiment with stride values
   - Compare Wood Wide percentiles (90-99)

3. **Analyze Performance:**
   - Note false positive rates
   - Check alerts by activity
   - Validate against known ground truth

4. **Deploy for Production:**
   - If results look good, integrate into your workflow
   - Use real Wood Wide API for production
   - Set up monitoring and logging

## Summary

The file upload feature makes the Health Sync Monitor dashboard fully self-contained:

- ✅ No preprocessing scripts needed
- ✅ Upload CSV → Get results immediately
- ✅ Sample CSV provided
- ✅ Interactive parameter tuning
- ✅ Real-time detection

**Try it now:**
```bash
streamlit run app.py
# Select "Upload CSV" → Download sample → Upload → Preprocess → Detect!
```
