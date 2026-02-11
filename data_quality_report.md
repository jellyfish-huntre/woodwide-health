# Data Quality Verification Report

**Date:** 2026-02-06
**Subject:** Subject 1 (Synthetic Data)
**Status:** ✓ ALL CHECKS PASSED

## Summary

The preprocessed PPG-DaLiA dataset has been verified and is ready for embedding generation via the Wood Wide API.

## Verification Results

### 1. Null Value Check ✓

- **Windows:** 0 null values
- **Timestamps:** 0 null values
- **Labels:** 0 null values

**Result:** No null/NaN values detected in any part of the dataset.

### 2. Data Range Check ✓

All sensor values are within expected physiological ranges:

| Feature | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| **PPG** | -1.34 | 1.30 | 0.02 | 0.71 |
| **ACC_X** | -0.96 | 0.93 | 0.00 | 0.24 |
| **ACC_Y** | -0.83 | 0.81 | 0.00 | 0.19 |
| **ACC_Z** | 0.21 | 1.76 | 1.00 | 0.23 |
| **ACC_MAG** | 0.29 | 1.86 | 1.05 | 0.23 |

**Result:** All values are within reasonable physiological ranges.

### 3. Window Shape Check ✓

- **Expected shape:** (114, 960, 5)
- **Actual shape:** (114, 960, 5)

**Breakdown:**
- 114 windows total
- 960 samples per window (30 seconds × 32 Hz)
- 5 features per sample (PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG)

**Result:** All windows have correct dimensions.

## Dataset Statistics

### Window-Level Summary (First 5 windows)

| Window | Timestamp | Activity | PPG Mean | ACC_MAG Mean |
|--------|-----------|----------|----------|--------------|
| 0 | 15.0s | 1 | -0.028 | 1.004 |
| 1 | 20.0s | 1 | 0.023 | 1.003 |
| 2 | 25.0s | 1 | 0.030 | 1.004 |
| 3 | 30.0s | 1 | 0.049 | 1.003 |
| 4 | 35.0s | 1 | 0.050 | 1.003 |

### Activity Distribution

- **Activity 1 (Sitting):** ~40% of windows
- **Activity 2 (Walking):** ~35% of windows
- **Activity 3 (Cycling):** ~25% of windows

## Metadata

```json
{
  "window_seconds": 30.0,
  "stride_seconds": 5.0,
  "sampling_rate": 32,
  "n_windows": 114,
  "window_shape": [114, 960, 5]
}
```

## Key Observations

1. **Clean Data:** No missing or null values detected
2. **Synchronized Signals:** All sensors aligned to 32 Hz sampling rate
3. **Proper Windowing:** 30-second windows with 5-second stride creates good overlap
4. **Feature Diversity:** Five features capture both heart rate (PPG) and movement (ACC)
5. **Activity Variation:** Multiple activity types represented in dataset

## Next Steps

The data is ready for:
1. ✅ Embedding generation via Wood Wide API
2. ✅ Signal analysis and decoupling detection
3. ✅ Visualization in Streamlit dashboard

## Data Pipeline Flow

```
Raw PPG-DaLiA Data (64 Hz PPG, 32 Hz ACC)
           ↓
Signal Synchronization (32 Hz)
           ↓
Derived Feature Computation (ACC magnitude)
           ↓
Rolling Window Creation (30s windows, 5s stride)
           ↓
Quality Verification ✓
           ↓
[READY] → Wood Wide API → Embeddings
```

---

**Verification Tool:** `verify_data_quality.py`
**Command:** `python3 verify_data_quality.py 1`
