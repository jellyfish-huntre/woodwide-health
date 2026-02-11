# Wood Wide Embedding-Based Detector Guide

## Overview

The Wood Wide detector uses multivariate embeddings to detect signal decoupling between heart rate and physical activity. Unlike traditional threshold-based approaches, it understands context and achieves dramatically lower false positive rates.

## Key Concept

### The Context Problem

Traditional fitness trackers cannot distinguish:
- ✓ High HR during exercise (normal)
- ⚠️ High HR during rest (concerning)

Both trigger alerts → 100% false positive rate during exercise

### The Wood Wide Solution

Embeddings capture the **relationship** between HR and activity:

```python
# Traditional approach
if hr > 100:
    alert()  # Can't tell if exercise or problem

# Wood Wide approach
embedding = woodwide_api.generate_embedding(window)
distance = ||embedding - normal_centroid||
if distance > threshold:
    alert()  # Only alerts when signals are decoupled
```

## How It Works

### 1. Learning Phase (fit)

The detector learns what "normal" looks like during exercise:

```python
from src.detectors.woodwide import WoodWideDetector

detector = WoodWideDetector(threshold_percentile=95)
detector.fit(embeddings, labels)
```

**What happens:**
1. Extract embeddings from exercise windows (cycling, walking, stairs)
2. Compute **normal activity centroid** (mean of exercise embeddings)
3. Measure distances from training samples to centroid
4. Set threshold at 95th percentile of distances

### 2. Detection Phase (predict)

For new data, detect anomalies:

```python
alerts = detector.predict(embeddings)
```

**What happens:**
1. Compute distance from each embedding to normal centroid
2. Alert when distance > threshold
3. Far from normal = signals are decoupled

### 3. Distance Metric

**Euclidean Distance:**
```python
distance = ||embedding - normal_centroid||_2
         = sqrt(sum((embedding - centroid)^2))
```

- Small distance → Similar to normal (exercise with appropriate HR)
- Large distance → Different from normal (potential decoupling)

## Performance Results

### Subject 1 (Realistic Synthetic Data)

**Configuration:**
- Threshold Percentile: 95
- Embedding Dimension: 128
- Activities: Sitting, Cycling, Walking, Ascending stairs

**Results:**

| Metric | Baseline (Threshold) | Wood Wide (Embeddings) | Improvement |
|--------|---------------------|------------------------|-------------|
| **False Positive Rate** | 100.0% | **5.6%** | **94.4% ↓** |
| **Alerts During Exercise** | 71 / 71 | **4 / 71** | **67 fewer** |
| **Alerts During Rest** | 38 / 43 | 17 / 43 | - |
| **Total Alerts** | 109 | **21** | **88 fewer** |

### By Activity

| Activity | Type | Alerts | Alert Rate |
|----------|------|--------|------------|
| **Sitting** | Rest | 17/43 | 39.5% |
| **Cycling** | Exercise | 2/48 | **4.2%** |
| **Walking** | Exercise | 2/23 | **8.7%** |

**Key Finding:** Only 4 false alarms during 71 exercise windows (vs. 71 in baseline)

## Usage

### Basic Usage

```python
from src.detectors.woodwide import WoodWideDetector

# Create detector
detector = WoodWideDetector(threshold_percentile=95)

# Fit on training data
detector.fit(embeddings, labels)

# Predict on new data
alerts = detector.predict(embeddings)

# Or combine fit + predict
result = detector.fit_predict(embeddings, labels)
print(f"Alerts: {result.alerts.sum()}")
print(f"False positive rate: {result.metrics['false_positive_rate_pct']:.1f}%")
```

### Command Line

```bash
# Run Wood Wide detection
python woodwide_detection.py 1 --use-mock

# Compare with baseline
python woodwide_detection.py 1 --use-mock --compare-baseline 100

# Save visualization
python woodwide_detection.py 1 --use-mock --save-plot output.png

# Adjust sensitivity (lower = more alerts)
python woodwide_detection.py 1 --use-mock --threshold-percentile 90

# Use real Wood Wide API
python woodwide_detection.py 1  # Requires WOOD_WIDE_API_KEY in .env
```

### Complete Workflow

```python
from src.embeddings.generate import send_windows_to_woodwide
from src.detectors.woodwide import WoodWideDetector

# 1. Generate embeddings
embeddings, metadata = send_windows_to_woodwide(
    windows,
    batch_size=32,
    embedding_dim=128,
    use_mock=True
)

# 2. Fit detector
detector = WoodWideDetector(threshold_percentile=95)
result = detector.fit_predict(embeddings, labels)

# 3. Analyze results
print(f"False positive rate: {result.metrics['false_positive_rate_pct']:.1f}%")
print(f"Alerts during exercise: {result.metrics['alerts_during_exercise']}")
print(f"Alerts during rest: {result.metrics['alerts_during_rest']}")

# 4. Save detector for production
detector.save("detector.pkl")

# 5. Load detector later
loaded_detector = WoodWideDetector.load("detector.pkl")
new_alerts = loaded_detector.predict(new_embeddings)
```

## Configuration

### Threshold Percentile

Controls sensitivity of detection:

| Percentile | Sensitivity | Use Case |
|------------|-------------|----------|
| **90** | High (more alerts) | Catch more anomalies, higher FP rate |
| **95** | Medium (balanced) | **Recommended default** |
| **99** | Low (fewer alerts) | Only most extreme anomalies |

```python
# High sensitivity
detector = WoodWideDetector(threshold_percentile=90)

# Low sensitivity
detector = WoodWideDetector(threshold_percentile=99)
```

### Exercise Labels

Customize which activities are considered "exercise":

```python
# Default: Cycling, Walking, Ascending stairs, Descending stairs
detector.fit(embeddings, labels, exercise_labels=[2, 3, 4, 5])

# Custom: Only cycling and walking
detector.fit(embeddings, labels, exercise_labels=[2, 3])

# Include vacuum cleaning as exercise
detector.fit(embeddings, labels, exercise_labels=[2, 3, 4, 5, 6])
```

## Advanced: Multi-Centroid Detector

For even better performance, use activity-specific centroids:

```python
from src.detectors.woodwide import MultiCentroidDetector

# Create multi-centroid detector
detector = MultiCentroidDetector(threshold_percentile=95)

# Fit learns separate centroid for each activity
detector.fit(embeddings, labels)

# Predict using activity-specific thresholds
alerts = detector.predict(embeddings, labels)
```

**Benefits:**
- Separate "normal" for each activity type
- Activity-specific distance thresholds
- Even lower false positive rates

## Visualization

The detection plot shows:

1. **Distance from Normal Centroid**
   - Blue line: Distance over time
   - Red dashed: Threshold
   - Red X: Alerts

2. **Distance by Activity**
   - Shows which activities are close/far from normal
   - Threshold line for reference

3. **Activity Timeline**
   - Context: what user was doing when

4. **Alert Timeline**
   - Red shading when alert is active

## Understanding Results

### Good Results

```
False Positive Rate: 5.6%
Alerts During Exercise: 4 / 71
```

✓ Low FP rate means detector understands context
✓ Few alerts during exercise = minimal false alarms

### Tuning Needed

```
False Positive Rate: 30%
Alerts During Exercise: 20 / 71
```

⚠️ Threshold percentile may be too low
→ Try increasing to 97 or 99

```
False Positive Rate: 0%
Alerts During Rest: 0 / 43
```

⚠️ Threshold percentile may be too high
→ Try decreasing to 90 or 92

## Technical Details

### Normal Centroid

The centroid represents "typical" exercise embeddings:

```python
normal_centroid = exercise_embeddings.mean(axis=0)
# Shape: (embedding_dim,)
```

This is the **center** of the normal activity cluster in embedding space.

### Distance Computation

Euclidean distance in embedding space:

```python
distances = np.linalg.norm(embeddings - normal_centroid, axis=1)
# Shape: (n_samples,)
```

### Threshold Selection

Percentile-based threshold from training data:

```python
threshold = np.percentile(training_distances, threshold_percentile)
```

- 95th percentile: Alert on top 5% most unusual
- Adaptive to data distribution

### Detection Rule

```python
alerts = distances > threshold
```

Simple but effective:
- True (alert) = embedding far from normal
- False (no alert) = embedding close to normal

## Why This Works

### Embeddings Encode Context

Wood Wide embeddings capture relationships:
- High HR + High ACC → One region of embedding space
- High HR + Low ACC → Different region of embedding space

### Normal Centroid = Expected Pattern

During exercise:
- High HR is normal
- Embeddings cluster together
- Centroid = "this is what exercise looks like"

### Anomaly = Deviation

When signals decouple:
- High HR but low activity
- Embedding moves away from normal cluster
- Large distance triggers alert

## Comparison with Baseline

### Baseline Threshold Detection

```python
if hr > 100:
    alert()
```

**Problems:**
- No context awareness
- 100% false positive rate during exercise
- Alert fatigue
- Unusable in practice

### Wood Wide Detection

```python
distance = ||embedding - normal_centroid||
if distance > threshold:
    alert()
```

**Benefits:**
- ✓ Context-aware
- ✓ 5.6% false positive rate
- ✓ 94% reduction in false alarms
- ✓ Practical for real-world use

## Production Deployment

### 1. Train Detector

```python
# Collect training data with known normal patterns
detector = WoodWideDetector(threshold_percentile=95)
detector.fit(training_embeddings, training_labels)
detector.save("production_detector.pkl")
```

### 2. Load in Production

```python
detector = WoodWideDetector.load("production_detector.pkl")
```

### 3. Real-Time Detection

```python
# For each new window
embedding = woodwide_api.generate_embedding(window)
alert = detector.predict(embedding.reshape(1, -1))[0]

if alert:
    send_notification("Signal decoupling detected!")
```

### 4. Monitoring

Track performance metrics:
```python
# Log distances for monitoring
distance = np.linalg.norm(embedding - detector.normal_centroid)
log_metric("distance_from_normal", distance)

# Alert on drift
if distance > detector.distance_threshold * 2:
    log_warning("Extreme anomaly detected")
```

## Files

- **Detector Implementation**: `src/detectors/woodwide.py`
- **Detection Script**: `woodwide_detection.py`
- **Results**: `data/woodwide_detection/`
- **Documentation**: `docs/WOODWIDE_DETECTOR_GUIDE.md`

## References

- **Baseline Detection**: `docs/BASELINE_DETECTION_RESULTS.md`
- **API Client**: `docs/API_CLIENT_GUIDE.md`
- **Embedding Generation**: `docs/EMBEDDING_GENERATION.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`

## Next Steps

1. ✅ Baseline threshold detection
2. ✅ Wood Wide embedding-based detection
3. ⏳ Build Streamlit dashboard for visualization
4. ⏳ Implement real-time streaming
5. ⏳ Deploy to production

## Summary

The Wood Wide detector achieves a **94% reduction in false alarms** compared to traditional threshold-based detection by understanding the context of heart rate in relation to physical activity. This makes it practical for real-world health monitoring applications.
