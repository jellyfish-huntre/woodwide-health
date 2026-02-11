"""
Code snippet utilities for documentation-style Streamlit dashboard.

Provides professional code display functions for the Health Sync Monitor dashboard.
"""

import streamlit as st
from typing import Optional
from pathlib import Path


def display_code_snippet(
    code: str,
    language: str = "python",
    caption: Optional[str] = None,
    file_reference: Optional[str] = None,
    line_numbers: bool = False
) -> None:
    """
    Display code snippet with professional styling.

    Args:
        code: Code string to display
        language: Programming language for syntax highlighting
        caption: Optional caption above code block
        file_reference: Optional file path reference (e.g., "src/embeddings/api_client.py:40-100")
        line_numbers: Show line numbers (default: False for cleaner look)
    """
    st.markdown('<div class="code-block-container">', unsafe_allow_html=True)

    if caption:
        st.markdown(f'<div class="code-caption">{caption}</div>', unsafe_allow_html=True)

    if file_reference:
        st.markdown(f'<div class="code-caption">📄 {file_reference}</div>', unsafe_allow_html=True)

    st.code(code, language=language, line_numbers=line_numbers)

    st.markdown('</div>', unsafe_allow_html=True)


def load_code_from_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """
    Load code snippet from file.

    Args:
        file_path: Path to file (relative to project root)
        start_line: Start line (1-indexed)
        end_line: End line (inclusive)

    Returns:
        Code string
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if start_line and end_line:
            lines = lines[start_line-1:end_line]
        elif start_line:
            lines = lines[start_line-1:]
        elif end_line:
            lines = lines[:end_line]

        return ''.join(lines)
    except FileNotFoundError:
        return f"# Error: File not found: {file_path}"
    except Exception as e:
        return f"# Error loading file: {e}"


def display_file_snippet(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    caption: Optional[str] = None
) -> None:
    """
    Load and display code snippet from file.

    Args:
        file_path: Path to source file
        start_line: Start line number
        end_line: End line number
        caption: Optional caption
    """
    code = load_code_from_file(file_path, start_line, end_line)

    # Create file reference
    if start_line and end_line:
        file_ref = f"{file_path}:{start_line}-{end_line}"
    else:
        file_ref = file_path

    display_code_snippet(
        code,
        language="python",
        caption=caption,
        file_reference=file_ref
    )


def create_callout(
    text: str,
    type: str = "info",
    title: Optional[str] = None
) -> None:
    """
    Create styled callout box.

    Args:
        text: Callout text (supports markdown)
        type: Type of callout (info, warning, success, error)
        title: Optional title
    """
    callout_classes = {
        "info": "info-callout",
        "warning": "warning-callout",
        "success": "success-callout",
        "error": "warning-callout"  # Reuse warning style
    }

    callout_class = callout_classes.get(type, "info-callout")

    html = f'<div class="{callout_class}">'
    if title:
        html += f'<strong>{title}</strong><br><br>'
    html += text
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)


def create_expandable_code(
    title: str,
    code: str,
    language: str = "python",
    expanded: bool = False
) -> None:
    """
    Create expandable code section.

    Args:
        title: Expander title
        code: Code to display
        language: Programming language
        expanded: Whether to start expanded
    """
    with st.expander(title, expanded=expanded):
        st.code(code, language=language, line_numbers=False)


# Predefined code snippets for common examples
CODE_SNIPPETS = {
    "threshold_detection_simple": """def detect_threshold(heart_rate: float, threshold: float = 100) -> bool:
    \"\"\"
    Naive threshold detection.
    Problem: Cannot distinguish exercise from health concerns.
    \"\"\"
    return heart_rate > threshold""",

    "woodwide_quickstart": """from src.embeddings.generate import send_windows_to_woodwide
from src.detectors.woodwide import WoodWideDetector

# Generate embeddings from time-series windows
embeddings, metadata = send_windows_to_woodwide(
    windows,
    batch_size=32,
    embedding_dim=128
)

# Train detector and detect anomalies
detector = WoodWideDetector(threshold_percentile=95)
result = detector.fit_predict(embeddings, labels)

# Access results
alerts = result.alerts  # Boolean array
distances = result.distances  # Distance from normal centroid""",

    "api_authentication": """from src.embeddings.api_client import APIClient
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize client (reads WOOD_WIDE_API_KEY from environment)
client = APIClient()

# Verify connection
health = client.check_health()
print(f"API Status: {health['status']}")""",

    "generate_embeddings_full": """from src.embeddings.generate import send_windows_to_woodwide

# Prepare windowed data
# Expected shape: (n_windows, window_length, n_features)
# Features: [PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG]

embeddings, metadata = send_windows_to_woodwide(
    windows,
    batch_size=32,          # Process 32 windows per request
    embedding_dim=128,      # 128-dimensional embeddings
    validate_input=True,    # Validate data quality
    use_mock=False          # Use real API
)

# Output shape: (n_windows, 128)
print(f"Generated {len(embeddings)} embeddings")
print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
print(f"Throughput: {metadata['windows_per_second']:.1f} windows/sec")""",

    "fit_detector": """from src.detectors.woodwide import WoodWideDetector

# Initialize detector
detector = WoodWideDetector(
    threshold_percentile=95,  # Alert on top 5% most unusual
    min_samples_for_fit=10    # Minimum training samples
)

# Learn normal activity patterns from exercise embeddings
detector.fit(
    embeddings=embeddings,
    labels=activity_labels,
    exercise_labels=[2, 3, 4, 5]  # Cycling, Walking, Stairs
)

# Detector learns:
# 1. Normal centroid (mean of exercise embeddings)
# 2. Distance threshold (95th percentile of training distances)

print(f"Normal centroid shape: {detector.normal_centroid.shape}")
print(f"Distance threshold: {detector.distance_threshold:.3f}")""",

    "predict_anomalies": """# Predict on new data
alerts, distances = detector.predict(
    embeddings,
    return_distances=True
)

# Interpretation:
# - alerts[i] == True: Window i shows signal decoupling
# - distances[i]: Euclidean distance from normal centroid
# - distances[i] > threshold: Signals are decoupled

# Example: Find first anomaly
anomaly_indices = np.where(alerts)[0]
if len(anomaly_indices) > 0:
    first_anomaly = anomaly_indices[0]
    print(f"First anomaly at window {first_anomaly}")
    print(f"Distance: {distances[first_anomaly]:.3f}")
    print(f"Threshold: {detector.distance_threshold:.3f}")""",

    "save_load_detector": """# Save trained detector for production use
detector.save('models/woodwide_detector.pkl')

# Load in production
from src.detectors.woodwide import WoodWideDetector
detector = WoodWideDetector.load('models/woodwide_detector.pkl')

# Process streaming data
new_embeddings = client.generate_embeddings(new_windows)
alerts = detector.predict(new_embeddings)""",

    "comparison_workflow": """# Complete comparison workflow
from src.embeddings.generate import send_windows_to_woodwide
from src.detectors.woodwide import WoodWideDetector

# Load preprocessed data
windows = data['windows']
labels = data['labels']

# Method 1: Baseline threshold
hr_bpm = extract_heart_rate(windows)
baseline_alerts = hr_bpm > threshold

# Method 2: Wood Wide embeddings
embeddings, _ = send_windows_to_woodwide(windows)
detector = WoodWideDetector(threshold_percentile=95)
result = detector.fit_predict(embeddings, labels)

# Compare metrics
is_exercise = np.isin(labels, [2, 3, 4, 5])
baseline_fp_rate = (baseline_alerts & is_exercise).sum() / is_exercise.sum() * 100
woodwide_fp_rate = result.metrics['false_positive_rate_pct']

print(f"Baseline FP rate: {baseline_fp_rate:.1f}%")
print(f"Wood Wide FP rate: {woodwide_fp_rate:.1f}%")
print(f"Improvement: {baseline_fp_rate - woodwide_fp_rate:.1f}%")"""
}


def display_predefined_snippet(snippet_key: str) -> None:
    """
    Display predefined code snippet.

    Args:
        snippet_key: Key from CODE_SNIPPETS dictionary
    """
    if snippet_key not in CODE_SNIPPETS:
        st.error(f"Unknown snippet key: {snippet_key}")
        return

    code = CODE_SNIPPETS[snippet_key]
    display_code_snippet(code, language="python")
