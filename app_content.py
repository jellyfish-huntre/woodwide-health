"""
Educational content strings for documentation-style dashboard.

Contains professionally-written content for the Health Sync Monitor dashboard.
"""

# Section introductions for each tab
SECTION_INTROS = {
    "overview": """**What you'll learn:**
- How Wood Wide embeddings solve the context problem in health monitoring
- The limitations of threshold-based detection
- Production-ready implementation patterns

**Prerequisites:**
- Python 3.8+
- Wood Wide API key (set in `.env`)
- Preprocessed health data (PPG + accelerometer)""",

    "baseline": """This section demonstrates why simple threshold detection fails for health monitoring.
You'll see how threshold methods produce high false positive rates during exercise.""",

    "woodwide": """This section demonstrates how Wood Wide embeddings enable context-aware detection
by learning the relationship between heart rate and physical activity.

**Prerequisites:**
- Preprocessed time-series windows (PPG + accelerometer)
- Wood Wide API key configured
- Understanding of embedding concepts (helpful but not required)""",

    "comparison": """This section compares threshold-based and embedding-based detection across key
performance metrics."""
}

# Algorithm explanations
ALGORITHM_EXPLANATIONS = {
    "threshold_problem": """Threshold detection cannot distinguish between:
- **Scenario A:** HR=120 BPM during cycling → Normal
- **Scenario B:** HR=120 BPM during sleep → Concerning

Both scenarios trigger identical alerts, resulting in 80-100% false positive rates
during exercise periods.""",

    "woodwide_how_it_works": """### Training Phase

The detector learns what "normal activity" looks like by computing a centroid:

```python
# Pseudocode
exercise_embeddings = embeddings[is_exercise]
normal_centroid = mean(exercise_embeddings)
distances = ||exercise_embeddings - normal_centroid||_2
threshold = percentile(distances, 95)
```

### Detection Phase

For each new window, compute distance from the normal centroid:

```python
# Pseudocode
for window_embedding in test_embeddings:
    distance = ||window_embedding - normal_centroid||_2
    if distance > threshold:
        alert = True  # Decoupling detected
```

### Why This Works

Embeddings encode contextual relationships between signals:
- **High HR + High activity** → Embedding near normal centroid → No alert
- **High HR + Low activity** → Embedding far from centroid → Alert

This is fundamentally different from threshold methods that examine signals independently.""",

    "interpretation_guide": """**How to interpret the chart:**
- **Blue line (accelerometer):** Physical activity level
- **Red line (heart rate):** Cardiac response
- **Red shading:** Detected signal decoupling

**Normal pattern:** HR and activity rise/fall together

**Anomaly pattern:** HR elevated while activity low (signals decoupled)"""
}

# API setup and configuration guides
API_SETUP_GUIDE = """Create a `.env` file in your project root:

```bash
WOOD_WIDE_API_KEY=your_api_key_here
```

**Important:** Never commit your API key to version control. Add `.env` to `.gitignore`."""

BATCHING_BEST_PRACTICES = """**Recommended batch sizes:**
- Standard datasets: `batch_size=32` (default)
- Large datasets (>1000 windows): `batch_size=16` to reduce memory usage
- Very large datasets (>5000 windows): `batch_size=8`

**Rate limiting:** The client handles retries automatically with exponential backoff. You don't need to implement your own retry logic."""

DEPLOYMENT_CONSIDERATIONS = """**Performance characteristics:**
- Embedding generation: ~0.1s per 32 windows
- Detection: <0.001s per window (CPU-only, no GPU required)
- Memory: ~4KB per embedding (128 dimensions × 4 bytes float32)

**Scaling recommendations:**
- For real-time processing: Generate embeddings in batches of 32-64 windows
- For batch processing: Use larger batches (128-256) for better throughput
- For edge deployment: Consider caching embeddings to minimize API calls"""

# Tutorial step descriptions
TUTORIAL_STEPS = {
    "step1_auth": """The first step is authenticating with the Wood Wide API. The client reads your API key from the environment and sets up secure HTTPS connections with automatic retry logic.""",

    "step2_embed": """This step transforms your time-series windows into dense vector embeddings. The API processes windows in batches for efficiency and returns unit-normalized embeddings.""",

    "step3_fit": """During the fitting phase, the detector learns what "normal" activity looks like by computing a centroid from exercise embeddings. This centroid represents the expected relationship between heart rate and physical activity.""",

    "step4_predict": """The prediction phase computes the Euclidean distance between each window's embedding and the normal centroid. Large distances indicate signal decoupling (e.g., high HR without high activity)."""
}

# Conclusion and key takeaways
CONCLUSIONS = {
    "baseline_limitations": """**Why threshold detection fails:**

1. **Context-blind:** Examines heart rate in isolation
2. **High false positive rate:** 80-100% during exercise
3. **Unusable for continuous monitoring:** Alert fatigue causes users to disable systems
4. **No learning:** Cannot adapt to individual patterns""",

    "iforest_limitations": """**Why Isolation Forest has moderate performance:**

1. **Hand-crafted features:** Requires domain expertise to engineer features
2. **Better than baseline:** Improves over simple thresholds
3. **Still context-limited:** Features don't capture signal relationships
4. **Configuration-sensitive:** Performance depends on contamination parameter""",

    "woodwide_advantages": """**Why Wood Wide succeeds:**

1. **Context-aware:** Understands HR-activity relationship
2. **Low false positive rate:** <10% during exercise
3. **Practical for deployment:** Manageable alert rate
4. **Embedding-based:** Learns relationships from data""",

    "key_insight": """### The Context Problem: Solved

Multivariate embeddings enable context-aware detection by encoding signal relationships in latent space. This fundamental capability cannot be achieved with threshold-based methods **or** hand-crafted features, regardless of optimization.

Wood Wide doesn't just set a better threshold or engineer better features—it solves a fundamentally different problem by understanding how signals relate to each other in a learned representation space.""",

    "three_way_insight": """### Three-Way Comparison: Lessons Learned

**Threshold Detection (Worst):** Simple but unusable due to excessive false positives

**Isolation Forest (Moderate):** Improvement over baseline but still limited by hand-crafted features

**Wood Wide (Best):** Context-aware embeddings capture signal relationships that hand-crafted features miss"""
}

# Further reading and references
FURTHER_READING = {
    "documentation": """**Documentation:**
- API Client Guide: `docs/API_CLIENT_GUIDE.md`
- Detector Implementation: `src/detectors/woodwide.py`
- Preprocessing Pipeline: `src/ingestion/preprocess.py`""",

    "examples": """**Code Examples:**
- Complete Examples: `examples/embedding_workflow.py`
- Baseline Detection: `baseline_threshold_detection.py`
- Wood Wide Detection: `woodwide_detection.py`""",

    "research": """**Research & Background:**
- Isolation Forest Comparison: `docs/ISOLATION_FOREST_BASELINE.md`
- Three-Way Comparison: `THREE_WAY_COMPARISON_SUMMARY.md`"""
}

# Callout messages for different scenarios
CALLOUT_MESSAGES = {
    "alert_fatigue": """Traditional threshold methods produce false positive rates of 80-100% during exercise, leading to alert fatigue that causes users to disable monitoring systems entirely.""",

    "context_awareness": """Wood Wide successfully distinguishes between normal exercise patterns and genuine anomalies by understanding signal relationships in embedding space.""",

    "production_ready": """This implementation is production-ready with proper error handling, rate limiting, retry logic, and performance optimization.""",

    "significant_improvement": """Wood Wide achieves a dramatic reduction in false positive rate compared to threshold detection. This improvement makes the system practical for continuous monitoring applications."""
}
