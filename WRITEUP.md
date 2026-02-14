# Health Sync Monitor: Context-Aware Anomaly Detection with Wood Wide AI

## Problem Framing

Wearable health monitors track heart rate and other measures to provide cardiac alerts. To provide valuable insights while avoiding alert fatigue, numeric reasoning must have high accuracy. Traditional methods often have to choose between producing large amounts of false positives, or missing out on genuine anomalies.

The core challenge is context-blindness. A heart rate of 120 BPM means very different things depending on whether someone is cycling or sleeping. Traditional threshold-based detection treats these identically.

We chose the **PPG-DaLiA dataset** (wrist-worn photoplethysmography and accelerometer data from 15 subjects across 8 activities) because it demonstrates non-trivial numeric reasoning: real physiological signals during both exercise and rest, where simple thresholds fail.

## Approach

### Dataset

PPG-DaLiA provides synchronized wrist signals:
- **PPG (Blood Volume Pulse):** 64 Hz, proxy for heart rate
- **Accelerometer (3-axis):** 32 Hz, physical activity context

We preprocessed the data by synchronizing to 32 Hz, computing accelerometer magnitude, and creating 30-second rolling windows with a 5-second stride. Each window is a data point: `(960 samples x 5 features)`.

### Three Detection Methods

**1. Heart Rate Threshold (baseline):**
Estimate heart rate via PPG peak detection, alert when HR exceeds a fixed threshold. This approach is simple but context-blind. Any threshold faces an inherent tradeoff between catching rest anomalies and triggering exercise false alarms.

**2. Isolation Forest (statistical baseline):**
Extract summary statistics per window (mean, std, min, max per channel) and fit an Isolation Forest on exercise windows to learn "normal" patterns. Isolation forest performs better than thresholds but is limited to hand-crafted features that can't capture temporal signal relationships.

**3. Wood Wide Embeddings (proposed):**
Extract per-window summary statistics and send them to the Wood Wide API, which returns 128-dimensional embeddings encoding multivariate relationships. A centroid-based detector in embedding space identifies anomalies as points far from the learned "normal exercise" centroid.

### Architecture

```
PPG-DaLiA  -->  Preprocessing  -->  Summary Features  -->  Wood Wide API  -->  Centroid Detector
(raw signals)   (synchronize,      (mean, std, min,      (128-dim           (Euclidean distance
                 window @ 32Hz)     max, percentiles,      embeddings)         from exercise
                                    cross-correlation)                         centroid)
```

## Results

Evaluated across 5 subjects (~740 minutes total, 8,894 windows).

### The Tradeoff Problem

Threshold detection faces a deadly tradeoff between sensitivity and specificity:

| Threshold | Exercise FP Rate | Rest Detection Rate |
|-----------|-----------------|-------------------|
| 70 BPM    | 82.4%           | 57.8%             |
| 80 BPM    | 62.3%           | 26.5%             |
| 90 BPM    | 25.4%           | 10.6%             |
| 100 BPM   | 5.8%            | 2.6%              |

To detect ~10% of rest anomalies (90 BPM threshold), you accept 25% exercise false positives.

### Wood Wide Breaks the Tradeoff

| Method                | Exercise FP Rate | Rest Detection Rate |
|-----------------------|-----------------|-------------------|
| Threshold (90 BPM)    | 25.4%           | 10.6%             |
| **Wood Wide (95th %ile)** | **5.1%**     | **9.2%**          |

Wood Wide achieves comparable rest detection (9.2% vs 10.6%) with 5x fewer false positives** (5.1% vs 25.4%). It breaks the threshold tradeoff by understanding that a high heart rate during exercise is normal.

### Per-Activity Breakdown (Subject 1)

| Activity        | Type     | Wood Wide Alert Rate |
|----------------|----------|---------------------|
| Ascending stairs | Exercise | 0.0%               |
| Walking         | Exercise | 2.0%                |
| Table soccer    | Exercise | 8.7%                |
| Cycling         | Exercise | 11.0%               |
| Driving         | Rest     | 0.6%                |
| Working         | Rest     | 1.7%                |
| Lunch break     | Rest     | 9.3%                |
| Sitting         | Rest     | 47.5%               |

Note: stairs and walking have near-zero false positive rates. Cycling is higher (11%), likely because it involves less movement variation than walking.

## Limitations

1. **Sitting false alerts (35-48%):** The centroid-based approach flags sitting windows at a high rate. This is because "sitting" embeddings are intentionally distant from the exercise centroid. The detector conflates "different from exercise" with "anomalous." A multi-centroid approach (one per activity) would address this.

2. **Medical data:** We define "false positive" as alerting in normal exercise contexts and "true detection" as alerting during rest and abnormal exercise. Real clinical anomalies (arrhythmia, tachycardia during sleep) would require labeled medical data.

3. **Summary feature extraction:** The Wood Wide API works best with tabular data, so we extract statistical summaries per window rather than sending raw time series. This process loses the temporal structure that could be informative. (The extraction is vectorized using bulk NumPy operations for performance.)

4. **Single centroid model:** One centroid cannot capture the full distribution of "normal." The existing `MultiCentroidDetector` class provides per-activity centroids but requires labeled data at inference time.

5. **PPG signal quality:** Wrist-worn PPG can be noisy, and heart rate estimates from peak detection are approximate. The embedding approach can better adapt since it operates on statistical features rather than individual peaks.

## What I Would Improve Next

1. **Multi-centroid detection:** Train separate centroids per activity type to reduce rest false alerts. The `MultiCentroidDetector` is already implemented in `src/detectors/woodwide.py`, but it requires activity labels at inference time.


2. **Cross-subject evaluation:** Train on subjects 1-3, test on subjects 4-5 to measure generalization. Currently, each subject is evaluated independently.

3. **Real-time streaming:** Implement sliding window processing for live data, with a cached model to avoid retraining.

## Dataset Citation

Reiss, A. (2019). PPG-DaLiA. UCI Machine Learning Repository. https://doi.org/10.24432/C5N312

Reiss, A. (2019). PPG-DaLiA. UCI Machine Learning Repository. https://doi.org/10.24432/C5N312
