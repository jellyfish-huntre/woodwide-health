# Heart Rate vs. Accelerometer Visualization Guide

## Overview

The Heart Rate vs. Accelerometer Magnitude chart is a dual-axis visualization that clearly demonstrates **why Wood Wide's context-aware detection succeeds** where traditional threshold methods fail.

## What It Shows

### Dual Y-Axes

**Left Axis (Red):** Heart Rate in BPM
- Extracted from PPG signal using peak detection
- Shows cardiovascular activity over time
- Higher values = faster heart rate

**Right Axis (Blue):** Accelerometer Magnitude in m/s²
- Computed as √(ACC_X² + ACC_Y² + ACC_Z²)
- Shows physical activity intensity
- Higher values = more movement

### Red Shaded Areas

**Anomaly Regions** - Wood Wide-detected signal decoupling
- Indicates where HR and activity relationship is unusual
- Represents potential health concerns
- Only appears when signals are truly decoupled

## How to Interpret

### Normal Patterns (No Red Shading)

**Scenario 1: Exercise**
```
High HR (↑) + High ACC (↑) = No anomaly ✅
Example: Cycling - HR 130 BPM, ACC 2.5 m/s²
Interpretation: Normal exercise response
```

**Scenario 2: Rest**
```
Low HR (↓) + Low ACC (↓) = No anomaly ✅
Example: Sitting - HR 70 BPM, ACC 0.5 m/s²
Interpretation: Normal resting state
```

### Anomaly Patterns (Red Shading)

**Scenario 3: Signal Decoupling**
```
High HR (↑) + Low ACC (↓) = ANOMALY ⚠️
Example: Sitting - HR 110 BPM, ACC 0.5 m/s²
Interpretation: Elevated HR without physical cause
```

This is the **context problem** that Wood Wide solves!

## Visual Examples

### Example 1: Successful Context Understanding

```
Time: 0-3 minutes (Cycling)
━━━━━━━━━━━━━━━━━━━━━━━━
Red Line (HR):    ↗️ Rising to 130 BPM
Blue Line (ACC):  ↗️ Rising to 2.5 m/s²
Red Shading:      ❌ None
━━━━━━━━━━━━━━━━━━━━━━━━
Interpretation: High HR during exercise is NORMAL
Wood Wide: No alert (understands context)
Baseline: Would alert (can't understand context)
```

### Example 2: Anomaly Detection

```
Time: 5-6 minutes (Sitting)
━━━━━━━━━━━━━━━━━━━━━━━━
Red Line (HR):    📈 Elevated at 110 BPM
Blue Line (ACC):  📉 Low at 0.6 m/s²
Red Shading:      🟥 YES
━━━━━━━━━━━━━━━━━━━━━━━━
Interpretation: High HR without activity is UNUSUAL
Wood Wide: Alert (detects decoupling)
Baseline: Might not alert (threshold may miss it)
```

## Technical Details

### Chart Creation

```python
fig = create_hr_vs_acceleration_chart(
    windows=windows,              # Data windows
    timestamps=timestamps,        # Window timestamps
    hr_bpm=hr_bpm,               # Heart rate array
    woodwide_alerts=alerts,      # Boolean anomaly array
    title="HR vs. ACC"
)
```

### Anomaly Shading Algorithm

```python
# Find continuous anomaly regions
for i, is_alert in enumerate(woodwide_alerts):
    if is_alert and not in_anomaly:
        anomaly_starts.append(i)  # Start of red region
        in_anomaly = True
    elif not is_alert and in_anomaly:
        anomaly_ends.append(i-1)  # End of red region
        in_anomaly = False

# Add shaded rectangles
for start, end in zip(anomaly_starts, anomaly_ends):
    fig.add_vrect(
        x0=time_minutes[start],
        x1=time_minutes[end],
        fillcolor="red",
        opacity=0.2
    )
```

### Accelerometer Magnitude Computation

```python
# ACC_MAG is the 5th feature (index 4)
acc_mag = windows[:, :, 4].mean(axis=1)
# Average ACC_MAG over each 30s window
```

### Heart Rate Extraction

```python
# Peak detection on PPG signal
from scipy.signal import find_peaks

for ppg_window in ppg_windows:
    peaks, _ = find_peaks(ppg_window, distance=int(32 * 0.4))
    if len(peaks) > 1:
        intervals = np.diff(peaks) / 32.0  # seconds
        hr_bpm = 60.0 / intervals.mean()   # BPM
```

## Dashboard Integration

### Wood Wide Tab

**Location:** Tab 3 → "Heart Rate vs. Physical Activity"

**Purpose:** Show how Wood Wide understands the HR-activity relationship

**Features:**
- Dual-axis plot
- Red anomaly shading
- Interpretation guide below chart
- Shows only Wood Wide anomalies

### Comparison Tab

**Location:** Tab 4 → "Signal Relationship Analysis"

**Purpose:** Demonstrate why Wood Wide outperforms baseline

**Features:**
- Same dual-axis plot
- Anomaly shading
- Side-by-side interpretation boxes (Normal vs. Anomaly)
- Contextual explanation

## Interactive Features

### Hover Information

**On Heart Rate Line:**
```
HR: 125.3 BPM
Time: 2.5 min
```

**On Accelerometer Line:**
```
ACC: 2.34 m/s²
Time: 2.5 min
```

### Zoom & Pan

- **Zoom:** Click and drag to zoom into specific time regions
- **Pan:** Shift + drag to pan across timeline
- **Reset:** Double-click to reset view
- **Legend:** Click legend items to show/hide traces

### Synchronized Hover

- Hover over any point to see both HR and ACC values at that time
- Unified tooltip shows data from both axes
- Easy comparison of signal relationships

## Use Cases

### 1. Demonstration

**Scenario:** Explaining the context problem

**Steps:**
1. Point to exercise period (high HR + high ACC)
2. Note: No red shading = Wood Wide understands this is normal
3. Point to rest period with elevated HR
4. Note: Red shading = Wood Wide detects anomaly
5. Compare to baseline which would alert on all high HR

### 2. Analysis

**Scenario:** Understanding detection results

**Questions to answer:**
- When did anomalies occur?
- What was the activity level during anomalies?
- How does HR correlate with ACC?
- Are anomalies truly unusual?

**Process:**
1. Scan chart for red shaded regions
2. Check ACC level during those times
3. Verify low ACC + high HR = true anomaly
4. Validate against activity labels

### 3. Tuning

**Scenario:** Adjusting Wood Wide threshold

**Process:**
1. Run detection with different percentiles
2. Observe changes in red shaded regions
3. Check if anomalies make sense (low ACC + high HR)
4. Tune until satisfactory balance

## Comparison with Baseline

### Traditional Threshold Chart

```
Heart Rate
  |     THRESHOLD ----
  |    /\    /\
  |   /  \  /  \     ← Would alert here (exercise)
  |  /    \/    \
  |_________________
```

**Problem:** Alerts during exercise (false positives)

### Wood Wide Dual-Axis Chart

```
Heart Rate
  |    /\    /\      ← No shading (normal)
  |   /  \  /  \
  |  /    \/    \
  |_________________
ACC Magnitude
  |    /\    /\      ← High ACC (exercise)
  |   /  \  /  \
  |  /    \/    \
  |_________________
```

**Solution:** No alert during exercise (understands context)

## Benefits

### Visual Clarity

- ✅ **Dual-axis** shows relationship at a glance
- ✅ **Color-coded** lines (red = HR, blue = ACC)
- ✅ **Red shading** makes anomalies obvious
- ✅ **Interactive** for detailed exploration

### Educational Value

- ✅ **Demonstrates** why context matters
- ✅ **Shows** what "decoupling" means
- ✅ **Proves** Wood Wide understands relationships
- ✅ **Contrasts** with threshold approach

### Analysis Power

- ✅ **Identify** when anomalies occur
- ✅ **Verify** anomalies are genuine
- ✅ **Understand** signal relationships
- ✅ **Tune** detection parameters

## Customization

### Change Colors

```python
# Heart rate line
line=dict(color='#e74c3c', width=2.5)  # Red

# Accelerometer line
line=dict(color='#3498db', width=2.5)  # Blue

# Anomaly shading
fillcolor="red", opacity=0.2
```

### Adjust Axes

```python
# Change axis titles
fig.update_yaxes(title_text="Custom HR Title", secondary_y=False)
fig.update_yaxes(title_text="Custom ACC Title", secondary_y=True)

# Change axis ranges
fig.update_yaxes(range=[50, 150], secondary_y=False)  # HR
fig.update_yaxes(range=[0, 5], secondary_y=True)      # ACC
```

### Modify Shading

```python
# More prominent shading
fillcolor="red", opacity=0.4

# Different color
fillcolor="orange", opacity=0.3

# Add border
line_width=2, line_color="red"
```

## Troubleshooting

### No Red Shading Appears

**Cause:** No anomalies detected

**Check:**
- Wood Wide threshold percentile (try lowering to 90)
- Verify anomalies exist: `result.alerts.any()`
- Check data has variation in HR and ACC

### Lines Overlap

**Cause:** Different scales need adjustment

**Solution:**
- Dual y-axes automatically handle this
- Adjust axis ranges if needed
- Use different line widths for clarity

### Chart Too Crowded

**Cause:** Too many data points

**Solution:**
- Zoom into specific time regions
- Use smaller time window
- Increase stride during preprocessing

## Summary

The Heart Rate vs. Accelerometer Magnitude chart is a powerful visualization that:

✅ **Demonstrates** the context problem visually
✅ **Shows** Wood Wide's understanding of signal relationships
✅ **Highlights** anomalies with red shading
✅ **Proves** why embeddings outperform thresholds
✅ **Enables** detailed analysis and tuning

**Key Insight:** By plotting both signals together, it becomes immediately obvious when they're **coupled** (normal) vs. **decoupled** (anomaly).

This is the visual proof of Wood Wide's superiority!
