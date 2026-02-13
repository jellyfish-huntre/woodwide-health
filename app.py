"""
Health Sync Monitor - Streamlit Dashboard

Interactive dashboard demonstrating Wood Wide AI's context-aware health monitoring
versus traditional threshold-based detection.

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import lzma
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Optional, Tuple, Union
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Bridge Streamlit Cloud secrets → os.environ so existing getenv() calls work
try:
    for key, value in st.secrets.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value
except FileNotFoundError:
    pass  # No secrets.toml — running locally with .env

from src.detectors.woodwide import WoodWideDetector, DetectionResult
from src.detectors.isolation_forest_detector import IsolationForestDetector
from src.embeddings.api_client import MockAPIClient
from src.embeddings.generate import send_windows_to_woodwide
from src.ingestion.preprocess import PPGDaLiaPreprocessor
from streamlit_helpers import (
    compute_isolation_forest_metrics,
    create_three_way_comparison_chart,
    create_three_way_timeline,
    create_comparison_table
)
from app_code_snippets import (
    display_code_snippet,
    display_file_snippet,
    display_predefined_snippet,
    create_callout,
    create_expandable_code,
    CODE_SNIPPETS
)
from app_content import (
    SECTION_INTROS,
    ALGORITHM_EXPLANATIONS,
    API_SETUP_GUIDE,
    BATCHING_BEST_PRACTICES,
    DEPLOYMENT_CONSIDERATIONS,
    TUTORIAL_STEPS,
    CONCLUSIONS,
    FURTHER_READING,
    CALLOUT_MESSAGES
)
import io

# Page config
st.set_page_config(
    page_title="Health Sync Monitor | Wood Wide AI",
    page_icon="static/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WoodWide.ai Dark Theme CSS - Professional Documentation Design
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global dark theme */
    .main {
        background-color: #0a0a0a;
        color: #F3F1E5;
    }

    /* Typography - WoodWide Style */
    .doc-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #F3F1E5;
        margin-bottom: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #262626;
    }

    .doc-subtitle {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.125rem;
        color: #8B7E6B;
        margin-bottom: 2.5rem;
        line-height: 1.7;
        font-weight: 400;
    }

    .section-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #F3F1E5;
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        padding-left: 1rem;
        border-left: 4px solid #EB9D6C;
    }

    /* Code blocks - Dark theme matching WoodWide */
    .code-block-container {
        background: #1a1a1a;
        border: 1px solid #262626;
        border-radius: 0.5rem;
        padding: 1.25rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3);
    }

    .code-caption {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.75rem;
        color: #8B7E6B;
        margin-bottom: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Override Streamlit code block styles for dark theme */
    .stCodeBlock {
        background: transparent !important;
    }

    .stCodeBlock code {
        font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', monospace !important;
        font-size: 0.875rem !important;
        line-height: 1.6 !important;
        color: #F3F1E5 !important;
        background: transparent !important;
    }

    /* Callouts - Dark theme */
    .info-callout {
        background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 0.5rem;
        color: #F3F1E5;
    }

    .info-callout strong {
        color: #60a5fa;
        font-weight: 600;
    }

    .warning-callout {
        background: linear-gradient(135deg, #2d1f0a 0%, #3d2a0f 100%);
        border-left: 4px solid #EB9D6C;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 0.5rem;
        color: #F3F1E5;
    }

    .warning-callout strong {
        color: #EB9D6C;
        font-weight: 600;
    }

    .success-callout {
        background: linear-gradient(135deg, #1a221a 0%, #1e2a1e 100%);
        border-left: 4px solid #F3F1E5;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 0.5rem;
        color: #F3F1E5;
    }

    .success-callout strong {
        color: #F3F1E5;
        font-weight: 600;
    }

    .note-callout {
        background: linear-gradient(135deg, #1a2a2a 0%, #1e3a3a 100%);
        border-left: 4px solid #8B7E6B;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 0.5rem;
        color: #F3F1E5;
    }

    .note-callout strong {
        color: #8B7E6B;
        font-weight: 600;
    }

    /* Tab descriptions - Dark */
    .tab-description {
        background: #1a1a1a;
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        font-size: 0.95rem;
        color: #8B7E6B;
        border-left: 4px solid #EB9D6C;
        line-height: 1.7;
    }

    /* Sidebar styling - Dark matching WoodWide */
    [data-testid="stSidebar"] {
        background: #0f0f0f;
        border-right: 1px solid #262626;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F3F1E5;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #8B7E6B;
    }

    /* Tabs styling - Orange accent like WoodWide */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #8B7E6B;
        padding: 0.75rem 1.5rem;
        border-radius: 0.375rem;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #EB9D6C 0%, #F0B58A 100%) !important;
        color: #0a0a0a !important;
    }

    /* Buttons - Orange accent */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 0.5rem;
        transition: all 0.2s;
        background-color: #1a1a1a;
        color: #F3F1E5;
        border: 1px solid #262626;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #EB9D6C 0%, #F0B58A 100%);
        border: none;
        color: #0a0a0a;
    }

    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #D68A5C 0%, #EB9D6C 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(235, 157, 108, 0.4);
    }

    .stButton button:hover {
        background-color: #262626;
        border-color: #404040;
    }

    /* Expanders - Dark theme */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #F3F1E5;
        background-color: #1a1a1a;
        border-radius: 0.5rem;
        border: 1px solid #262626;
    }

    .streamlit-expanderHeader:hover {
        background-color: #262626;
    }

    /* Metrics - Dark cards */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #F3F1E5;
    }

    [data-testid="stMetricLabel"] {
        color: #8B7E6B;
    }

    /* Links - Orange accent */
    a {
        color: #EB9D6C;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }

    a:hover {
        color: #F0B58A;
        text-decoration: underline;
    }

    /* Text elements */
    .stMarkdown {
        color: #F3F1E5;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #F3F1E5;
    }

    /* Dataframes - Dark theme */
    .dataframe {
        background-color: #1a1a1a;
        color: #F3F1E5;
    }

    /* Input widgets - Dark theme */
    .stTextInput input,
    .stSelectbox select,
    .stSlider {
        background-color: #1a1a1a;
        color: #F3F1E5;
        border: 1px solid #262626;
    }

    /* Dividers */
    hr {
        border-color: #262626;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .doc-header {
            font-size: 1.75rem;
        }

        .doc-subtitle {
            font-size: 1rem;
        }

        .section-header {
            font-size: 1.375rem;
        }

        .code-block-container {
            padding: 1rem;
            overflow-x: auto;
        }
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Selection color - Orange */
    ::selection {
        background-color: #EB9D6C;
        color: #0a0a0a;
    }

    /* Info boxes */
    .stInfo {
        background-color: #1a1a1a;
        border: 1px solid #262626;
        color: #8B7E6B;
    }

    .stSuccess {
        background-color: #1a221a;
        border: 1px solid #F3F1E5;
        color: #F3F1E5;
    }

    .stWarning {
        background-color: #3d2a0f;
        border: 1px solid #EB9D6C;
        color: #EB9D6C;
    }

    .stError {
        background-color: #3d1a1a;
        border: 1px solid #ef4444;
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar and app logo
st.logo(
    "static/69600c52653fc6e72def1c7b_Logo white.svg",
    icon_image="static/69600c52653fc6e72def1c7b_Logo white.svg",
)


# Helper functions
def format_subject_id(subject_id) -> str:
    """Format subject_id for file paths, handling both int and str."""
    if isinstance(subject_id, str):
        return subject_id
    else:
        return f"{subject_id:02d}"


# Data loading functions
@st.cache_data
def load_subject_data(subject_id: Union[int, str]) -> Optional[Dict]:
    """Load preprocessed subject data."""
    subject_str = format_subject_id(subject_id)
    data_path = Path(f"data/processed/subject_{subject_str}_processed.pkl")
    if not data_path.exists():
        return None

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@st.cache_data
def load_embeddings(subject_id: Union[int, str]) -> Optional[Tuple[np.ndarray, Dict]]:
    """Load cached embeddings if available."""
    subject_str = format_subject_id(subject_id)
    embeddings_file = Path(f"data/embeddings/subject_{subject_str}_embeddings.npy")
    metadata_file = Path(f"data/embeddings/subject_{subject_str}_metadata.pkl")

    if not embeddings_file.exists() or not metadata_file.exists():
        return None

    embeddings = np.load(embeddings_file)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    return embeddings, metadata


@st.cache_data
def load_baseline_results(subject_id: Union[int, str], threshold: int = 100) -> Optional[Dict]:
    """Load baseline detection results."""
    subject_str = format_subject_id(subject_id)
    results_file = Path(f"data/baseline_detection/subject_{subject_str}_threshold_{threshold}.pkl")
    if not results_file.exists():
        return None

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


@st.cache_data
def load_woodwide_results(subject_id: Union[int, str]) -> Optional[Dict]:
    """Load Wood Wide detection results."""
    subject_str = format_subject_id(subject_id)
    results_file = Path(f"data/woodwide_detection/subject_{subject_str}_results.pkl")
    if not results_file.exists():
        return None

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


@st.cache_data
def load_isolation_forest_results(subject_id: Union[int, str]) -> Optional[Dict]:
    """Load Isolation Forest detection results."""
    subject_str = format_subject_id(subject_id)
    results_file = Path(f"data/isolation_forest_detection/subject_{subject_str}_results.pkl")
    if not results_file.exists():
        return None

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def _decompress_bundled_data() -> bool:
    """Decompress LZMA-compressed data from bundled_data/ to data/processed/."""
    bundled = Path(__file__).parent / "bundled_data"
    if not bundled.exists():
        return False
    compressed_files = sorted(bundled.glob("*.pkl.lzma"))
    if not compressed_files:
        return False
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Decompress only the first subject to stay within Streamlit Cloud disk limits
    cf = compressed_files[0]
    out = processed_dir / cf.stem  # strips .lzma → .pkl
    if not out.exists():
        out.write_bytes(lzma.decompress(cf.read_bytes()))
    return True


def _clear_uploaded_data():
    """Remove persisted uploaded data from disk and clear caches."""
    for f in [
        Path("data/processed/subject_uploaded_processed.pkl"),
        Path("data/embeddings/subject_uploaded_embeddings.npy"),
        Path("data/embeddings/subject_uploaded_metadata.pkl"),
    ]:
        if f.exists():
            f.unlink()
    load_subject_data.clear()
    load_embeddings.clear()
    st.session_state.uploaded_data = None


def generate_sample_csv() -> str:
    """Generate a sample PPG-DaLiA CSV for download with realistic patterns.

    Creates 10 minutes of data with multiple activities to ensure sufficient
    exercise samples for Wood Wide detector training.

    Returns:
        CSV string
    """
    # Create 10 minutes of sample data (matches realistic synthetic data)
    duration = 600  # seconds (10 minutes)
    ppg_rate = 64
    acc_rate = 32

    # Activity schedule (in seconds)
    # Ensure sufficient exercise data for detector training
    activities = [
        (0, 60, 1),      # 0-1 min: Sitting
        (60, 180, 4),    # 1-3 min: Cycling
        (180, 240, 1),   # 3-4 min: Sitting
        (240, 360, 7),   # 4-6 min: Walking
        (360, 420, 1),   # 6-7 min: Sitting
        (420, 540, 4),   # 7-9 min: Cycling
        (540, 600, 1),   # 9-10 min: Sitting
    ]

    # Generate timestamps
    t_ppg = np.arange(0, duration, 1/ppg_rate)
    t_acc = np.arange(0, duration, 1/acc_rate)

    # Initialize signals
    ppg = np.zeros(len(t_ppg))
    accx = np.zeros(len(t_acc))
    accy = np.zeros(len(t_acc))
    accz = np.zeros(len(t_acc))
    labels = np.zeros(len(t_acc), dtype=int)

    # Generate realistic signals for each activity period
    for start, end, activity_label in activities:
        # Indices for this period
        ppg_start_idx = int(start * ppg_rate)
        ppg_end_idx = int(end * ppg_rate)
        acc_start_idx = int(start * acc_rate)
        acc_end_idx = int(end * acc_rate)

        # Time arrays for this period
        t_ppg_period = t_ppg[ppg_start_idx:ppg_end_idx] - start
        t_acc_period = t_acc[acc_start_idx:acc_end_idx] - start

        # Activity-specific parameters
        if activity_label == 1:  # Sitting
            hr_bpm = 72 + np.random.randn() * 3  # Resting HR
            ppg_baseline = 0.0
            acc_intensity = 0.15  # Minimal movement
        elif activity_label == 4:  # Cycling
            hr_bpm = 125 + np.random.randn() * 5  # Elevated HR
            ppg_baseline = 0.0
            acc_intensity = 1.2  # Moderate movement
        elif activity_label == 7:  # Walking
            hr_bpm = 110 + np.random.randn() * 5  # Moderately elevated
            ppg_baseline = 0.0
            acc_intensity = 0.8  # Walking movement
        else:
            hr_bpm = 75
            ppg_baseline = 0.0
            acc_intensity = 0.2

        # Generate PPG signal with realistic heart rate
        hr_freq = hr_bpm / 60.0  # Convert BPM to Hz
        ppg_signal = ppg_baseline + np.sin(2 * np.pi * hr_freq * t_ppg_period)
        ppg_signal += 0.1 * np.sin(2 * np.pi * 2 * hr_freq * t_ppg_period)  # Harmonics
        ppg_signal += np.random.randn(len(t_ppg_period)) * 0.05  # Noise

        # Generate accelerometer with activity-specific patterns
        if activity_label == 4:  # Cycling - rhythmic pattern
            freq = 1.5  # ~90 RPM
            accx[acc_start_idx:acc_end_idx] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_period)
            accy[acc_start_idx:acc_end_idx] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_period + np.pi/2)
            accz[acc_start_idx:acc_end_idx] = 9.81 + acc_intensity * 0.5 * np.sin(2 * np.pi * freq * t_acc_period)
        elif activity_label == 7:  # Walking - step pattern
            freq = 1.8  # ~108 steps/min
            accx[acc_start_idx:acc_end_idx] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_period)
            accy[acc_start_idx:acc_end_idx] = acc_intensity * 0.6 * np.sin(2 * np.pi * freq * t_acc_period + np.pi/3)
            accz[acc_start_idx:acc_end_idx] = 9.81 + acc_intensity * np.sin(2 * np.pi * 2 * freq * t_acc_period)
        else:  # Sitting - minimal random movement
            accx[acc_start_idx:acc_end_idx] = acc_intensity * np.random.randn(len(t_acc_period))
            accy[acc_start_idx:acc_end_idx] = acc_intensity * np.random.randn(len(t_acc_period))
            accz[acc_start_idx:acc_end_idx] = 9.81 + acc_intensity * np.random.randn(len(t_acc_period))

        # Add noise to accelerometer
        accx[acc_start_idx:acc_end_idx] += np.random.randn(len(t_acc_period)) * 0.05
        accy[acc_start_idx:acc_end_idx] += np.random.randn(len(t_acc_period)) * 0.05
        accz[acc_start_idx:acc_end_idx] += np.random.randn(len(t_acc_period)) * 0.05

        # Assign PPG and labels
        ppg[ppg_start_idx:ppg_end_idx] = ppg_signal
        labels[acc_start_idx:acc_end_idx] = activity_label

    # Downsample PPG to match ACC rate for CSV
    from scipy.interpolate import interp1d
    ppg_interp = interp1d(t_ppg, ppg, kind='linear', bounds_error=False, fill_value='extrapolate')
    ppg_downsampled = ppg_interp(t_acc)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp_acc': t_acc,
        'ppg': ppg_downsampled,
        'accX': accx,
        'accY': accy,
        'accZ': accz,
        'label': labels
    })

    return df.to_csv(index=False)


def generate_demo_processed_data() -> Dict:
    """Generate demo processed data directly for Streamlit Cloud.

    Creates 10 minutes of synthetic PPG+ACC data with correct PPG-DaLiA
    activity labels, preprocesses it into windows, and returns the standard
    {windows, timestamps, labels, metadata} dict.
    """
    duration = 600  # 10 minutes
    ppg_rate = 64
    acc_rate = 32

    # Activity schedule with correct PPG-DaLiA labels
    activities = [
        (0, 60, 1),      # 0-1 min: Sitting
        (60, 180, 4),    # 1-3 min: Cycling
        (180, 240, 1),   # 3-4 min: Sitting
        (240, 360, 7),   # 4-6 min: Walking
        (360, 420, 1),   # 6-7 min: Sitting
        (420, 540, 4),   # 7-9 min: Cycling
        (540, 600, 1),   # 9-10 min: Sitting
    ]

    t_ppg = np.arange(0, duration, 1 / ppg_rate)
    t_acc = np.arange(0, duration, 1 / acc_rate)

    ppg = np.zeros(len(t_ppg))
    accx = np.zeros(len(t_acc))
    accy = np.zeros(len(t_acc))
    accz = np.zeros(len(t_acc))
    labels = np.zeros(len(t_acc), dtype=int)

    np.random.seed(42)

    for start, end, activity_label in activities:
        ppg_si = int(start * ppg_rate)
        ppg_ei = int(end * ppg_rate)
        acc_si = int(start * acc_rate)
        acc_ei = int(end * acc_rate)

        t_ppg_p = t_ppg[ppg_si:ppg_ei] - start
        t_acc_p = t_acc[acc_si:acc_ei] - start

        if activity_label == 1:  # Sitting
            hr_bpm = 72 + np.random.randn() * 3
            acc_intensity = 0.15
        elif activity_label == 4:  # Cycling
            hr_bpm = 125 + np.random.randn() * 5
            acc_intensity = 1.2
        elif activity_label == 7:  # Walking
            hr_bpm = 110 + np.random.randn() * 5
            acc_intensity = 0.8
        else:
            hr_bpm = 75
            acc_intensity = 0.2

        hr_freq = hr_bpm / 60.0
        ppg_signal = np.sin(2 * np.pi * hr_freq * t_ppg_p)
        ppg_signal += 0.1 * np.sin(2 * np.pi * 2 * hr_freq * t_ppg_p)
        ppg_signal += np.random.randn(len(t_ppg_p)) * 0.05

        if activity_label == 4:  # Cycling
            freq = 1.5
            accx[acc_si:acc_ei] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_p)
            accy[acc_si:acc_ei] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_p + np.pi / 2)
            accz[acc_si:acc_ei] = 9.81 + acc_intensity * 0.5 * np.sin(2 * np.pi * freq * t_acc_p)
        elif activity_label == 7:  # Walking
            freq = 1.8
            accx[acc_si:acc_ei] = acc_intensity * np.sin(2 * np.pi * freq * t_acc_p)
            accy[acc_si:acc_ei] = acc_intensity * 0.6 * np.sin(2 * np.pi * freq * t_acc_p + np.pi / 3)
            accz[acc_si:acc_ei] = 9.81 + acc_intensity * np.sin(2 * np.pi * 2 * freq * t_acc_p)
        else:  # Sitting
            accx[acc_si:acc_ei] = acc_intensity * np.random.randn(len(t_acc_p))
            accy[acc_si:acc_ei] = acc_intensity * np.random.randn(len(t_acc_p))
            accz[acc_si:acc_ei] = 9.81 + acc_intensity * np.random.randn(len(t_acc_p))

        accx[acc_si:acc_ei] += np.random.randn(len(t_acc_p)) * 0.05
        accy[acc_si:acc_ei] += np.random.randn(len(t_acc_p)) * 0.05
        accz[acc_si:acc_ei] += np.random.randn(len(t_acc_p)) * 0.05

        ppg[ppg_si:ppg_ei] = ppg_signal
        labels[acc_si:acc_ei] = activity_label

    # Feed directly into preprocessing pipeline (bypasses CSV round-trip)
    ppg_df = pd.DataFrame({'ppg': ppg, 'timestamp': t_ppg})
    acc_df = pd.DataFrame({
        'acc_x': accx, 'acc_y': accy, 'acc_z': accz, 'timestamp': t_acc
    })
    labels_df = pd.DataFrame({'activity': labels, 'timestamp': t_acc})

    preprocessor = PPGDaLiaPreprocessor()
    synced_df = preprocessor.synchronize_signals(ppg_df, acc_df, labels_df)
    synced_df = preprocessor.compute_derived_features(synced_df)
    windows, timestamps, window_labels = preprocessor.create_rolling_windows(
        synced_df, window_seconds=30.0, stride_seconds=5.0
    )

    return {
        'windows': windows,
        'timestamps': timestamps,
        'labels': window_labels,
        'metadata': {
            'n_windows': len(windows),
            'window_seconds': 30.0,
            'stride_seconds': 5.0,
            'sampling_rate': 32,
        }
    }


def parse_uploaded_csv(uploaded_file) -> Optional[Dict]:
    """Parse uploaded PPG-DaLiA CSV file.

    Expected format:
    - ppg: Column with PPG signal (64 Hz)
    - accX, accY, accZ: Accelerometer columns (32 Hz)
    - label: Activity label
    - timestamp_ppg, timestamp_acc: Timestamps

    Returns:
        Dictionary with parsed data or None if invalid
    """
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.lower()

        # Required columns
        required_cols = ['ppg', 'accx', 'accy', 'accz', 'label']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Need: {required_cols}")
            st.error(f"Found: {df.columns.tolist()}")
            return None

        # Extract data
        data = {
            'ppg': df['ppg'].values.astype(np.float32),
            'acc_x': df['accx'].values.astype(np.float32),
            'acc_y': df['accy'].values.astype(np.float32),
            'acc_z': df['accz'].values.astype(np.float32),
            'labels': df['label'].values.astype(int)
        }

        # Create timestamps if not present
        if 'timestamp_ppg' in df.columns:
            data['timestamps_ppg'] = df['timestamp_ppg'].values
        else:
            # Assume 64 Hz for PPG
            data['timestamps_ppg'] = np.arange(len(data['ppg'])) / 64.0

        if 'timestamp_acc' in df.columns:
            data['timestamps_acc'] = df['timestamp_acc'].values
        else:
            # Assume 32 Hz for ACC
            data['timestamps_acc'] = np.arange(len(data['acc_x'])) / 32.0

        # Add sampling rates
        data['ppg_sampling_rate'] = 64
        data['acc_sampling_rate'] = 32

        return data

    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None


def preprocess_uploaded_data(
    data: Dict,
    window_seconds: float = 30.0,
    stride_seconds: float = 5.0
) -> Optional[Dict]:
    """Preprocess uploaded data into windows.

    Args:
        data: Parsed data dictionary
        window_seconds: Window length in seconds
        stride_seconds: Stride between windows in seconds

    Returns:
        Dictionary with preprocessed windows or None if error
    """
    try:
        # Create preprocessor (no parameters needed)
        preprocessor = PPGDaLiaPreprocessor()

        # Convert raw arrays to DataFrames for preprocessor
        ppg_df = pd.DataFrame({
            'ppg': data['ppg'],
            'timestamp': data['timestamps_ppg']
        })

        acc_df = pd.DataFrame({
            'acc_x': data['acc_x'],
            'acc_y': data['acc_y'],
            'acc_z': data['acc_z'],
            'timestamp': data['timestamps_acc']
        })

        labels_df = pd.DataFrame({
            'activity': data['labels'],
            'timestamp': data['timestamps_acc']  # Use ACC timestamps for labels
        })

        # Synchronize signals to common timeline
        synced_df = preprocessor.synchronize_signals(ppg_df, acc_df, labels_df)

        # Compute derived features (accelerometer magnitude)
        synced_df = preprocessor.compute_derived_features(synced_df)

        # Create rolling windows (window_seconds and stride_seconds go here)
        windows, timestamps, labels = preprocessor.create_rolling_windows(
            synced_df,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds
        )

        return {
            'windows': windows,
            'timestamps': timestamps,
            'labels': labels,
            'metadata': {
                'n_windows': len(windows),
                'window_seconds': window_seconds,
                'stride_seconds': stride_seconds,
                'sampling_rate': 32
            }
        }

    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def extract_heart_rate_simple(windows: np.ndarray) -> np.ndarray:
    """Quick HR extraction for visualization."""
    from scipy.signal import find_peaks

    ppg_windows = windows[:, :, 0]
    sampling_rate = 32.0
    hr_bpm = np.zeros(len(ppg_windows))

    for i, ppg_window in enumerate(ppg_windows):
        peaks, _ = find_peaks(ppg_window, distance=int(sampling_rate * 0.4))

        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sampling_rate
            hr = 60.0 / peak_intervals.mean()
            hr_bpm[i] = hr
        else:
            hr_bpm[i] = 70.0 + ppg_window.std() * 30

    return hr_bpm


# Visualization functions
def create_activity_timeline(labels: np.ndarray, timestamps: np.ndarray, activity_map: Dict) -> go.Figure:
    """Create interactive activity timeline."""
    time_minutes = (timestamps - timestamps[0]) / 60

    # Create color mapping
    colors = px.colors.qualitative.Plotly
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(activity_map.keys())}

    fig = go.Figure()

    for label, activity in activity_map.items():
        mask = labels == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=time_minutes[mask],
                y=np.ones(mask.sum()) * label,
                mode='markers',
                name=activity,
                marker=dict(size=10, color=color_map[label]),
                hovertemplate=f'{activity}<br>Time: %{{x:.1f}} min<extra></extra>'
            ))

    fig.update_layout(
        title="Activity Timeline",
        xaxis_title="Time (minutes)",
        yaxis_title="Activity",
        yaxis=dict(
            tickmode='array',
            tickvals=list(activity_map.keys()),
            ticktext=list(activity_map.values())
        ),
        height=300,
        hovermode='closest'
    )

    return fig


def create_hr_vs_acceleration_chart(
    windows: np.ndarray,
    timestamps: np.ndarray,
    hr_bpm: np.ndarray,
    woodwide_alerts: Optional[np.ndarray] = None,
    title: str = "Heart Rate vs. Accelerometer Magnitude"
) -> go.Figure:
    """Create dual-axis chart showing HR vs. acceleration with anomaly shading.

    Args:
        windows: Data windows (n_windows, 960, 5) - features: [PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG]
        timestamps: Window timestamps
        hr_bpm: Heart rate in BPM
        woodwide_alerts: Boolean array of Wood Wide anomaly detections
        title: Chart title

    Returns:
        Plotly figure with dual y-axes
    """
    time_minutes = (timestamps - timestamps[0]) / 60

    # Extract accelerometer magnitude (last feature, index 4)
    acc_mag = windows[:, :, 4].mean(axis=1)  # Average ACC_MAG over each window

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add anomaly shading first (so it's in the background)
    if woodwide_alerts is not None and woodwide_alerts.any():
        # Find continuous anomaly regions
        anomaly_starts = []
        anomaly_ends = []

        in_anomaly = False
        for i, is_alert in enumerate(woodwide_alerts):
            if is_alert and not in_anomaly:
                # Start of anomaly region
                anomaly_starts.append(i)
                in_anomaly = True
            elif not is_alert and in_anomaly:
                # End of anomaly region
                anomaly_ends.append(i - 1)
                in_anomaly = False

        # Handle case where anomaly extends to end
        if in_anomaly:
            anomaly_ends.append(len(woodwide_alerts) - 1)

        # Add shaded regions for each anomaly
        for start, end in zip(anomaly_starts, anomaly_ends):
            fig.add_vrect(
                x0=time_minutes[start],
                x1=time_minutes[end],
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Anomaly" if start == anomaly_starts[0] else None,
                annotation_position="top left"
            )

    # Add heart rate trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=hr_bpm,
            mode='lines',
            name='Heart Rate',
            line=dict(color='#e74c3c', width=2.5),
            hovertemplate='HR: %{y:.1f} BPM<br>Time: %{x:.1f} min<extra></extra>'
        ),
        secondary_y=False
    )

    # Add accelerometer magnitude trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=acc_mag,
            mode='lines',
            name='Accelerometer Magnitude',
            line=dict(color='#3498db', width=2.5),
            hovertemplate='ACC: %{y:.2f} m/s²<br>Time: %{x:.1f} min<extra></extra>'
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="Time (minutes)")
    fig.update_yaxes(
        title_text="Heart Rate (BPM)",
        secondary_y=False,
        title_font=dict(color='#e74c3c'),
        tickfont=dict(color='#e74c3c')
    )
    fig.update_yaxes(
        title_text="Accelerometer Magnitude (m/s²)",
        secondary_y=True,
        title_font=dict(color='#3498db'),
        tickfont=dict(color='#3498db')
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, weight='bold')
        ),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240, 242, 246, 0.5)'
    )

    return fig


def create_detection_comparison(
    timestamps: np.ndarray,
    hr_bpm: np.ndarray,
    baseline_alerts: np.ndarray,
    woodwide_alerts: np.ndarray,
    woodwide_distances: Optional[np.ndarray],
    baseline_threshold: float,
    woodwide_threshold: Optional[float]
) -> go.Figure:
    """Create comparison plot of baseline vs Wood Wide detection."""
    time_minutes = (timestamps - timestamps[0]) / 60

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Heart Rate with Baseline Threshold",
            "Wood Wide Distance from Normal" if woodwide_distances is not None else "Wood Wide Alerts",
            "Alert Comparison"
        ),
        vertical_spacing=0.12,
        row_heights=[0.35, 0.35, 0.30]
    )

    # Row 1: Heart rate with baseline threshold
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=hr_bpm,
            mode='lines',
            name='Heart Rate',
            line=dict(color='blue', width=2),
            hovertemplate='HR: %{y:.1f} BPM<br>Time: %{x:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )

    # Baseline threshold line
    fig.add_hline(
        y=baseline_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {baseline_threshold} BPM",
        row=1, col=1
    )

    # Baseline alerts
    if baseline_alerts.any():
        fig.add_trace(
            go.Scatter(
                x=time_minutes[baseline_alerts],
                y=hr_bpm[baseline_alerts],
                mode='markers',
                name='Baseline Alerts',
                marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                hovertemplate='ALERT<br>HR: %{y:.1f} BPM<extra></extra>'
            ),
            row=1, col=1
        )

    # Row 2: Wood Wide distances or alerts
    if woodwide_distances is not None:
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=woodwide_distances,
                mode='lines',
                name='Distance',
                line=dict(color='green', width=2),
                hovertemplate='Distance: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )

        if woodwide_threshold is not None:
            fig.add_hline(
                y=woodwide_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {woodwide_threshold:.3f}",
                row=2, col=1
            )

        if woodwide_alerts.any():
            fig.add_trace(
                go.Scatter(
                    x=time_minutes[woodwide_alerts],
                    y=woodwide_distances[woodwide_alerts],
                    mode='markers',
                    name='Wood Wide Alerts',
                    marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                    hovertemplate='ALERT<br>Distance: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )

    # Row 3: Alert comparison
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=baseline_alerts.astype(float) * 2,
            mode='lines',
            name='Baseline Alerts',
            fill='tozeroy',
            line=dict(color='red', width=0),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='Baseline Alert<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=woodwide_alerts.astype(float),
            mode='lines',
            name='Wood Wide Alerts',
            fill='tozeroy',
            line=dict(color='green', width=0),
            fillcolor='rgba(0, 255, 0, 0.3)',
            hovertemplate='Wood Wide Alert<extra></extra>'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="HR (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="Distance", row=2, col=1)
    fig.update_yaxes(title_text="Alert", row=3, col=1, tickmode='array', tickvals=[0, 1, 2], ticktext=['None', 'Wood Wide', 'Baseline'])

    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def create_metrics_comparison(baseline_metrics: Dict, woodwide_metrics: Dict) -> go.Figure:
    """Create comparison bar chart of metrics."""
    metrics = ['False Positive Rate (%)', 'Alerts During Exercise', 'Total Alerts']
    baseline_values = [
        baseline_metrics['false_positive_rate_pct'],
        baseline_metrics['alerts_during_exercise'],
        baseline_metrics['total_alerts']
    ]
    woodwide_values = [
        woodwide_metrics['false_positive_rate_pct'],
        woodwide_metrics['alerts_during_exercise'],
        woodwide_metrics['total_alerts']
    ]

    fig = go.Figure(data=[
        go.Bar(name='Baseline (Threshold)', x=metrics, y=baseline_values, marker_color='red'),
        go.Bar(name='Wood Wide (Embeddings)', x=metrics, y=woodwide_values, marker_color='green')
    ])

    fig.update_layout(
        title="Detection Performance Comparison",
        barmode='group',
        height=400,
        yaxis_title="Value",
        hovermode='x unified'
    )

    return fig


# Main app
def main():
    # Header with WoodWide branding - load SVG logos
    import base64
    logo_white_path = Path("static/69600c52653fc6e72def1c7b_Logo white.svg")
    footer_wordmark_path = Path("static/695ec02e4b197a17d27fffc9_Footer Wordmark.svg")

    logo_white_b64 = ""
    if logo_white_path.exists():
        logo_white_b64 = base64.b64encode(logo_white_path.read_bytes()).decode()

    footer_wordmark_b64 = ""
    if footer_wordmark_path.exists():
        footer_wordmark_b64 = base64.b64encode(footer_wordmark_path.read_bytes()).decode()

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #262626;">
        <div style="display: flex; align-items: center;">
            <div style="display: flex; align-items: center; padding: 0.5rem 1rem; background: #1a1a1a; border-radius: 0.5rem; border: 1px solid #262626; margin-right: 1.5rem;">
                <img src="data:image/svg+xml;base64,{logo_white_b64}" alt="Wood Wide AI" style="height: 28px;" />
            </div>
            <div>
                <div class="doc-header" style="margin-bottom: 0; padding-bottom: 0; border-bottom: none;">Health Sync Monitor</div>
            </div>
        </div>
        <a href="#" style="display: inline-flex; align-items: center; padding: 0.625rem 1.25rem; background: linear-gradient(135deg, #EB9D6C 0%, #F0B58A 100%); color: #0a0a0a; text-decoration: none; border-radius: 0.5rem; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.875rem; transition: all 0.2s;">
            Dashboard →
        </a>
    </div>
    <div class="doc-subtitle">
        Context-aware health monitoring powered by Wood Wide AI multivariate embeddings
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Data source selection
        data_source = st.radio(
            "Data Source",
            options=["Pre-processed Data", "Upload CSV"],
            help="Choose between pre-processed subjects or upload your own CSV file"
        )

        st.divider()

        # Subject selection or file upload
        if data_source == "Pre-processed Data":
            available_subjects = []
            if Path("data/processed").exists():
                for p in Path("data/processed").glob("subject_*_processed.pkl"):
                    sid = p.stem.split('_')[1]
                    try:
                        available_subjects.append(int(sid))
                    except ValueError:
                        available_subjects.append(sid)

            if not available_subjects:
                # Try decompressing bundled real data first, then fall back to demo
                if 'data_init_failed' not in st.session_state:
                    with st.spinner("Loading data (first run)..."):
                        try:
                            if _decompress_bundled_data():
                                load_subject_data.clear()
                                load_embeddings.clear()
                                st.rerun()
                            else:
                                # No bundled data — generate synthetic demo
                                demo_data = generate_demo_processed_data()
                                processed_dir = Path("data/processed")
                                processed_dir.mkdir(parents=True, exist_ok=True)
                                with open(processed_dir / "subject_demo_processed.pkl", 'wb') as f:
                                    pickle.dump(demo_data, f)
                                mock_client = MockAPIClient(embedding_dim=128)
                                embeddings = mock_client.generate_embeddings(demo_data['windows'])
                                embeddings_dir = Path("data/embeddings")
                                embeddings_dir.mkdir(parents=True, exist_ok=True)
                                np.save(embeddings_dir / "subject_demo_embeddings.npy", embeddings)
                                with open(embeddings_dir / "subject_demo_metadata.pkl", 'wb') as f:
                                    pickle.dump({
                                        'subject_id': 'demo',
                                        'embeddings_shape': embeddings.shape,
                                        'timestamps': demo_data['timestamps'],
                                        'labels': demo_data['labels'],
                                        'window_metadata': demo_data['metadata'],
                                    }, f)
                                load_subject_data.clear()
                                load_embeddings.clear()
                                st.rerun()
                        except Exception as e:
                            st.session_state.data_init_failed = True
                            st.error(f"Data initialization failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                st.warning("No processed data found.")
                st.info("Switch to 'Upload CSV' to upload your own data.")
                return

            subject_id = st.selectbox(
                "Select Subject",
                options=available_subjects,
                format_func=lambda x: "Demo Data" if x == "demo" else f"Subject {x}"
            )

            # Load data
            data = load_subject_data(subject_id)
            if data is None:
                st.error(f"Could not load data for subject {subject_id}")
                return

        else:  # Upload CSV
            st.subheader("Upload PPG-DaLiA CSV")

            # Download sample CSV
            sample_csv = generate_sample_csv()
            st.download_button(
                label="Download Sample CSV",
                data=sample_csv,
                file_name="sample_ppg_dalia.csv",
                mime="text/csv",
                help="Download a sample CSV to see the expected format",
                use_container_width=True
            )

            st.divider()

            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="CSV with columns: ppg, accX, accY, accZ, label"
            )

            if uploaded_file is None:
                # Try to recover previously saved uploaded data from disk
                recovered_data = load_subject_data("uploaded")
                if recovered_data is not None:
                    data = recovered_data
                    st.session_state.uploaded_data = data
                    st.info(
                        f"Restored previously uploaded data "
                        f"({len(data['windows'])} windows). "
                        f"Upload a new CSV to replace it."
                    )
                    if st.button(
                        "Clear Uploaded Data",
                        help="Remove saved uploaded data and start fresh"
                    ):
                        _clear_uploaded_data()
                        st.rerun()
                else:
                    st.info("""
                    **Required CSV format:**
                    - `ppg` - PPG signal
                    - `accX`, `accY`, `accZ` - Accelerometer
                    - `label` - Activity label
                    - (Optional) `timestamp_acc`

                    Download the sample CSV above to see the format!
                    """)
                    return
            else:
                # Parse uploaded file
                with st.spinner("Parsing CSV file..."):
                    parsed_data = parse_uploaded_csv(uploaded_file)

                if parsed_data is None:
                    st.error("[ERROR] Failed to parse CSV file. Check format.")
                    return

                st.success(f"[SUCCESS] Loaded {len(parsed_data['ppg'])} PPG samples")

                # Preprocessing parameters
                st.subheader("Preprocessing")

                window_seconds = st.slider(
                    "Window Length (seconds)",
                    min_value=10,
                    max_value=60,
                    value=30,
                    step=5
                )

                stride_seconds = st.slider(
                    "Stride (seconds)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1
                )

                # Run preprocessing button
                if st.button("Preprocess Data", type="primary", use_container_width=True):
                    with st.spinner("Preprocessing data..."):
                        data = preprocess_uploaded_data(
                            parsed_data,
                            window_seconds=window_seconds,
                            stride_seconds=stride_seconds
                        )

                    if data is None:
                        st.error("[ERROR] Failed to preprocess data")
                        return

                    # Persist to disk so data survives page reloads
                    processed_dir = Path("data/processed")
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    with open(processed_dir / "subject_uploaded_processed.pkl", 'wb') as f:
                        pickle.dump({**data, 'subject_id': 'uploaded'}, f)
                    load_subject_data.clear()

                    st.session_state.uploaded_data = data
                    st.success(f"[SUCCESS] Created {len(data['windows'])} windows")
                    st.rerun()

                # Use preprocessed data from session state
                if st.session_state.uploaded_data is not None:
                    data = st.session_state.uploaded_data
                    st.info(f"Using {len(data['windows'])} preprocessed windows")
                else:
                    st.warning("Upload and preprocess data to continue")
                    return

            subject_id = "uploaded"

        st.divider()

        # Baseline threshold
        baseline_threshold = st.slider(
            "Baseline HR Threshold (BPM)",
            min_value=80,
            max_value=140,
            value=100,
            step=5,
            help="Heart rate threshold for baseline detection"
        )

        # Wood Wide threshold percentile
        woodwide_percentile = st.slider(
            "Wood Wide Threshold Percentile",
            min_value=85,
            max_value=99,
            value=95,
            step=1,
            help="Higher = fewer alerts (less sensitive)"
        )

        st.divider()

        # Generate embeddings option — default to mock when no API key
        api_key_configured = bool(
            os.getenv("WOOD_WIDE_API_KEY")
            and os.getenv("WOOD_WIDE_API_KEY") != "your_api_key_here"
        )
        use_mock = st.checkbox(
            "Use Mock API",
            value=not api_key_configured,
            help="Use mock API for embeddings (no API key needed)"
        )

        # Show API key status
        if not use_mock:
            api_key = os.getenv("WOOD_WIDE_API_KEY")
            if api_key and api_key != "your_api_key_here":
                st.success(f"[SUCCESS] API Key loaded (ends with ...{api_key[-4:]})")
            else:
                st.error("[ERROR] API Key not found. Set WOOD_WIDE_API_KEY in .env or Streamlit secrets")
                st.info("Enable 'Use Mock API' above to test without an API key")

        force_regenerate = st.checkbox(
            "Force Regenerate Embeddings",
            value=False,
            help="Regenerate embeddings even if cached"
        )

        st.divider()

        # Run Detection Button
        run_button = st.button(
            "Run Detection",
            type="primary",
            use_container_width=True,
            help="Run baseline and Wood Wide detection on the data"
        )

        if run_button:
            st.session_state.run_detection = True

        st.divider()

        # Info
        st.info("""
        **How it works:**

        1. **Baseline**: Simple HR threshold
        2. **Wood Wide**: Context-aware embeddings
        3. **Comparison**: Performance analysis

        Select configuration options above and click 'Run Detection' to begin analysis.
        """)

    # Extract data
    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']

    activity_map = {
        0: 'Transient',
        1: 'Sitting',
        2: 'Ascending stairs',
        3: 'Table soccer',
        4: 'Cycling',
        5: 'Driving',
        6: 'Lunch break',
        7: 'Walking',
        8: 'Working'
    }

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview & Quick Start",
        "Baseline: Classic Approaches",
        "Wood Wide: Embedding Detection",
        "Three-Way Comparison"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Quick Start Guide")

        # What you'll learn and Prerequisites
        st.markdown(SECTION_INTROS["overview"])

        st.divider()

        # Dataset Overview
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Subject ID", "Demo" if subject_id == "demo" else subject_id)
        with col2:
            st.metric("Windows", len(windows))
        with col3:
            duration = (timestamps[-1] - timestamps[0]) / 60
            st.metric("Duration", f"{duration:.1f} min")
        with col4:
            st.metric("Window Size", "30 sec")

        st.divider()

        # Activity distribution
        st.subheader("Activity Distribution")

        activity_counts = pd.DataFrame([
            {'Activity': activity_map[label], 'Count': (labels == label).sum()}
            for label in np.unique(labels)
        ])

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(activity_counts, hide_index=True, use_container_width=True)

        with col2:
            fig = px.pie(
                activity_counts,
                values='Count',
                names='Activity',
                title='Activity Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Activity timeline
        st.subheader("Activity Timeline")
        fig = create_activity_timeline(labels, timestamps, activity_map)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # The Context Problem
        st.subheader("The Context Problem")

        st.markdown("""
        Traditional fitness trackers use simple heart rate thresholds:
        """)

        # Code snippet showing threshold detection
        display_code_snippet(
            CODE_SNIPPETS["threshold_detection_simple"],
            caption="Threshold Detection (Baseline)"
        )

        st.markdown("""
        **The problem:** This approach cannot distinguish between:
        - High HR during exercise (normal)
        - High HR during rest (concerning)

        **Result:** High false positive rates during exercise
        """)

        # Alert fatigue callout
        create_callout(
            CALLOUT_MESSAGES["alert_fatigue"],
            type="warning",
            title="Alert Fatigue Problem"
        )

        st.divider()

        # The Wood Wide Solution
        st.subheader("The Wood Wide Solution")

        st.markdown("""
        Wood Wide embeddings capture the relationship between heart rate and activity context:
        """)

        # Code snippet showing Wood Wide solution
        display_code_snippet(
            CODE_SNIPPETS["woodwide_quickstart"],
            caption="Wood Wide Detection"
        )

        st.markdown("""
        **Result:** False positive rates <10% while maintaining sensitivity to genuine anomalies.
        """)

    # Tab 2: Baseline Detection
    with tab2:
        st.header("Baseline: Classic Approaches")
        st.markdown("### Understanding the limitations of threshold and statistical methods")

        # Tab description
        st.markdown(f'<div class="tab-description">{SECTION_INTROS["baseline"]}</div>', unsafe_allow_html=True)

        st.subheader("Method 1: Heart Rate Threshold")
        st.markdown(f"Alert when Heart Rate > {baseline_threshold} BPM")

        # Extract HR
        with st.spinner("Extracting heart rate from PPG signal..."):
            hr_bpm = extract_heart_rate_simple(windows)

        # Apply threshold
        baseline_alerts = hr_bpm > baseline_threshold

        # Compute metrics
        is_exercise = np.isin(labels, [2, 3, 4, 7])
        is_rest = ~is_exercise

        baseline_metrics = {
            'total_alerts': int(baseline_alerts.sum()),
            'alerts_during_exercise': int((baseline_alerts & is_exercise).sum()),
            'alerts_during_rest': int((baseline_alerts & is_rest).sum()),
            'false_positive_rate_pct': (baseline_alerts & is_exercise).sum() / is_exercise.sum() * 100 if is_exercise.sum() > 0 else 0,
            'exercise_windows': int(is_exercise.sum()),
            'rest_windows': int(is_rest.sum())
        }

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Alerts",
                baseline_metrics['total_alerts'],
                delta=f"{baseline_metrics['total_alerts']/len(windows)*100:.1f}%"
            )
        with col2:
            st.metric(
                "Alerts During Exercise",
                f"{baseline_metrics['alerts_during_exercise']}/{baseline_metrics['exercise_windows']}",
                delta=f"{baseline_metrics['false_positive_rate_pct']:.1f}% FP rate",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Alerts During Rest",
                f"{baseline_metrics['alerts_during_rest']}/{baseline_metrics['rest_windows']}"
            )
        with col4:
            st.metric(
                "HR Range",
                f"{hr_bpm.min():.0f}-{hr_bpm.max():.0f} BPM"
            )

        # Alert if high FP rate
        if baseline_metrics['false_positive_rate_pct'] > 50:
            create_callout(
                f"**High False Positive Rate: {baseline_metrics['false_positive_rate_pct']:.1f}%**\n\n"
                f"{baseline_metrics['alerts_during_exercise']} false alarms during exercise.\n\n"
                "Threshold-based detection produces excessive false alarms during exercise, "
                "leading to alert fatigue. Users disable monitoring systems with FP rates >50%.",
                type="warning",
                title="Warning: Unusable for Continuous Monitoring"
            )

        st.divider()

        # Visualization
        st.subheader("Detection Visualization")

        time_minutes = (timestamps - timestamps[0]) / 60

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Heart Rate Over Time", "Alert Timeline"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # HR plot
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=hr_bpm,
                mode='lines',
                name='Heart Rate',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        fig.add_hline(
            y=baseline_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {baseline_threshold} BPM",
            row=1, col=1
        )

        if baseline_alerts.any():
            fig.add_trace(
                go.Scatter(
                    x=time_minutes[baseline_alerts],
                    y=hr_bpm[baseline_alerts],
                    mode='markers',
                    name='Alerts',
                    marker=dict(size=12, color='red', symbol='x', line=dict(width=2))
                ),
                row=1, col=1
            )

        # Alert timeline
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=baseline_alerts.astype(float),
                mode='lines',
                name='Alert Active',
                fill='tozeroy',
                line=dict(color='red', width=0),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="HR (BPM)", row=1, col=1)
        fig.update_yaxes(title_text="Alert", row=2, col=1, tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])

        fig.update_layout(height=600, showlegend=True, hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # By activity
        st.subheader("Performance by Activity")

        activity_perf = []
        for label, activity in activity_map.items():
            mask = labels == label
            if mask.sum() > 0:
                n_alerts = (baseline_alerts & mask).sum()
                activity_perf.append({
                    'Activity': activity,
                    'Type': 'Exercise' if label in [2, 3, 4, 7] else 'Rest',
                    'Windows': mask.sum(),
                    'Alerts': n_alerts,
                    'Alert Rate (%)': n_alerts / mask.sum() * 100
                })

        df_perf = pd.DataFrame(activity_perf)
        st.dataframe(df_perf, hide_index=True, use_container_width=True)

        # ============================================================
        # ISOLATION FOREST SECTION
        # ============================================================
        st.divider()
        st.subheader("Method 2: Isolation Forest (Classic ML)")

        st.markdown("""
        Can a more sophisticated algorithm do better? **Isolation Forest** is a popular
        anomaly detection method that considers **multiple signals simultaneously**
        rather than just heart rate alone.
        """)

        with st.expander("How Isolation Forest Works", expanded=False):
            st.markdown(ALGORITHM_EXPLANATIONS["isolation_forest_how_it_works"])

        # Run Isolation Forest
        with st.spinner("Training Isolation Forest on exercise windows..."):
            iforest_detector = IsolationForestDetector(contamination=0.1)
            iforest_detector.fit(windows, labels)
            iforest_result = iforest_detector.predict(windows)

            iforest_metrics = compute_isolation_forest_metrics(
                if_alerts=iforest_result.alerts,
                labels=labels
            )
            iforest_metrics['alerts'] = iforest_result.alerts

        # Metrics row
        if_col1, if_col2, if_col3, if_col4 = st.columns(4)

        with if_col1:
            st.metric(
                "Total Alerts",
                iforest_metrics['total_alerts'],
                delta=f"{iforest_metrics['total_alerts']/len(windows)*100:.1f}%"
            )
        with if_col2:
            st.metric(
                "Alerts During Exercise",
                f"{iforest_metrics['alerts_during_exercise']}/{iforest_metrics['exercise_windows']}",
                delta=f"{iforest_metrics['false_positive_rate_pct']:.1f}% FP rate",
                delta_color="inverse"
            )
        with if_col3:
            st.metric(
                "Alerts During Rest",
                f"{iforest_metrics['alerts_during_rest']}/{iforest_metrics['rest_windows']}"
            )
        with if_col4:
            improvement = baseline_metrics['false_positive_rate_pct'] - iforest_metrics['false_positive_rate_pct']
            st.metric(
                "FP Improvement vs Threshold",
                f"{improvement:.1f}%",
                delta="better" if improvement > 0 else "worse",
                delta_color="normal" if improvement > 0 else "inverse"
            )

        if iforest_metrics['false_positive_rate_pct'] > 10:
            create_callout(
                f"**Exercise False Positive Rate: {iforest_metrics['false_positive_rate_pct']:.1f}%**\n\n"
                f"Better than threshold ({baseline_metrics['false_positive_rate_pct']:.1f}%), "
                "but still too high for reliable continuous monitoring.",
                type="warning",
                title="Improvement, But Not Enough"
            )

        st.divider()

        # Visualization: Anomaly Scores + Alert Timeline
        st.subheader("Isolation Forest Detection Visualization")

        # Negate scores so higher = more anomalous (more intuitive)
        negated_scores = -iforest_result.anomaly_scores
        negated_threshold = -iforest_result.threshold

        if_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Anomaly Score Over Time (higher = more unusual)", "Alert Timeline"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # Row 1: Anomaly scores
        if_fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=negated_scores,
                mode='lines',
                name='Anomaly Score',
                line=dict(color='#f39c12', width=2),
                hovertemplate='Score: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Decision threshold line
        if_fig.add_hline(
            y=negated_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Decision Threshold",
            row=1, col=1
        )

        # Highlight anomalous points
        if iforest_result.alerts.any():
            if_fig.add_trace(
                go.Scatter(
                    x=time_minutes[iforest_result.alerts],
                    y=negated_scores[iforest_result.alerts],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(size=8, color='red', symbol='x', line=dict(width=1.5)),
                    hovertemplate='Anomaly Score: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Row 2: Alert timeline
        if_fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=iforest_result.alerts.astype(float),
                mode='lines',
                name='Alert Active',
                fill='tozeroy',
                line=dict(color='#f39c12', width=0),
                fillcolor='rgba(243, 156, 18, 0.3)'
            ),
            row=2, col=1
        )

        if_fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        if_fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
        if_fig.update_yaxes(
            title_text="Alert", row=2, col=1,
            tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']
        )

        if_fig.update_layout(height=600, showlegend=True, hovermode='x unified')

        st.plotly_chart(if_fig, use_container_width=True)

        st.divider()

        # Performance by Activity
        st.subheader("Isolation Forest: Performance by Activity")

        if_activity_perf = []
        for label, activity in activity_map.items():
            mask = labels == label
            if mask.sum() > 0:
                n_alerts = (iforest_result.alerts & mask).sum()
                if_activity_perf.append({
                    'Activity': activity,
                    'Type': 'Exercise' if label in [2, 3, 4, 7] else 'Rest',
                    'Windows': int(mask.sum()),
                    'Alerts': int(n_alerts),
                    'Alert Rate (%)': n_alerts / mask.sum() * 100
                })

        df_if_perf = pd.DataFrame(if_activity_perf)
        st.dataframe(df_if_perf, hide_index=True, use_container_width=True)

        # ============================================================
        # QUICK COMPARISON & UNIFIED CONCLUSION
        # ============================================================
        st.divider()
        st.subheader("Threshold vs. Isolation Forest")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("**Threshold Detection**")
            st.metric("Exercise FP Rate",
                      f"{baseline_metrics['false_positive_rate_pct']:.1f}%")
            st.metric("Total Alerts", baseline_metrics['total_alerts'])
            st.markdown("*Single signal (HR only), no learning*")

        with comp_col2:
            st.markdown("**Isolation Forest**")
            st.metric("Exercise FP Rate",
                      f"{iforest_metrics['false_positive_rate_pct']:.1f}%")
            st.metric("Total Alerts", iforest_metrics['total_alerts'])
            st.markdown("*Multi-signal, hand-crafted features*")

        st.divider()

        # Unified conclusion
        st.subheader("Why Classic Methods Fail")

        create_callout(
            ALGORITHM_EXPLANATIONS["threshold_problem"],
            type="info",
            title="Issue 1: Context Blindness (Threshold)"
        )

        create_callout(
            CONCLUSIONS["iforest_limitations"],
            type="warning",
            title="Issue 2: Feature Limitations (Isolation Forest)"
        )

        create_callout(
            "**Neither approach captures signal relationships.** "
            "Threshold detection examines HR in isolation. "
            "Isolation Forest uses hand-crafted features (mean/std) that cannot encode "
            "the temporal coupling between heart rate and physical activity. "
            "To solve this, we need **learned representations** that understand context. "
            "See the next tab for Wood Wide's embedding-based approach.",
            type="info",
            title="The Path Forward: Context-Aware Embeddings"
        )

    # Tab 3: Wood Wide Detection
    with tab3:
        st.header("Wood Wide: Embedding-Based Detection")
        st.markdown("### Context-aware anomaly detection using multivariate embeddings")

        # Tab description
        st.markdown(f'<div class="tab-description">{SECTION_INTROS["woodwide"]}</div>', unsafe_allow_html=True)

        # Tutorial steps with code snippets
        st.subheader("Step 1: API Authentication")
        st.markdown(TUTORIAL_STEPS["step1_auth"])

        display_code_snippet(
            CODE_SNIPPETS["api_authentication"],
            caption="Initialize Wood Wide API Client"
        )

        create_callout(API_SETUP_GUIDE, type="info", title="API Key Setup")

        st.divider()

        st.subheader("Step 2: Generate Embeddings")
        st.markdown(TUTORIAL_STEPS["step2_embed"])

        display_code_snippet(
            CODE_SNIPPETS["generate_embeddings_full"],
            caption="Generate Embeddings from Time-Series Windows"
        )

        create_callout(BATCHING_BEST_PRACTICES, type="warning", title="Batching Best Practices")

        st.divider()

        st.subheader("Step 3: Fit Detector")
        st.markdown(TUTORIAL_STEPS["step3_fit"])

        display_code_snippet(
            CODE_SNIPPETS["fit_detector"],
            caption="Fit Wood Wide Detector"
        )

        st.divider()

        st.subheader("Step 4: Detect Anomalies")
        st.markdown(TUTORIAL_STEPS["step4_predict"])

        display_code_snippet(
            CODE_SNIPPETS["predict_anomalies"],
            caption="Predict Anomalies"
        )

        st.divider()

        st.subheader("Detection Results")

        # Load or generate embeddings
        embeddings_data = load_embeddings(subject_id)

        if embeddings_data is None or force_regenerate:
            with st.status("Generating embeddings via Wood Wide API...", expanded=True) as status:
                try:
                    # Single placeholder that gets overwritten for poll updates
                    current_step = st.empty()

                    def on_progress(step, message):
                        if step == "poll_status":
                            # Overwrite in place so poll lines don't stack
                            current_step.markdown(f"`{message}`")
                        elif step == "poll_done":
                            current_step.empty()
                            st.markdown(f"**{message}**")
                        elif step == "done":
                            st.markdown(f"**{message}**")
                        elif step in ("upload_done", "train_done"):
                            st.caption(message)
                        else:
                            st.text(message)

                    embeddings, emb_metadata = send_windows_to_woodwide(
                        windows,
                        batch_size=32,
                        embedding_dim=128,
                        use_mock=use_mock,
                        progress_callback=on_progress
                    )

                    # Save
                    embeddings_dir = Path("data/embeddings")
                    embeddings_dir.mkdir(parents=True, exist_ok=True)
                    subject_str = format_subject_id(subject_id)
                    np.save(embeddings_dir / f"subject_{subject_str}_embeddings.npy", embeddings)
                    with open(embeddings_dir / f"subject_{subject_str}_metadata.pkl", 'wb') as f:
                        pickle.dump(emb_metadata, f)

                    # Invalidate stale cache so load_embeddings() picks up the new files
                    load_embeddings.clear()

                    status.update(label="Embeddings generated!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Embedding generation failed", state="error")
                    st.error(f"[ERROR] Failed to generate embeddings: {e}")
                    return
        else:
            embeddings, emb_metadata = embeddings_data
            st.success("[SUCCESS] Loaded cached embeddings")

        # Store in session state so Tab 4 can access without cache issues
        st.session_state.embeddings_cache[subject_id] = (embeddings, emb_metadata)

        # Fit detector
        with st.spinner("Fitting Wood Wide detector..."):
            detector = WoodWideDetector(threshold_percentile=woodwide_percentile)
            result = detector.fit_predict(embeddings, labels)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Alerts",
                result.metrics['total_alerts'],
                delta=f"{result.metrics['total_alerts']/len(windows)*100:.1f}%"
            )
        with col2:
            st.metric(
                "Alerts During Exercise",
                f"{result.metrics['alerts_during_exercise']}/{result.metrics['exercise_windows']}",
                delta=f"{result.metrics['false_positive_rate_pct']:.1f}% FP rate",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Alerts During Rest",
                f"{result.metrics['alerts_during_rest']}/{result.metrics['rest_windows']}"
            )
        with col4:
            st.metric(
                "Distance Threshold",
                f"{result.threshold:.3f}"
            )

        # Success message if low FP rate
        if result.metrics['false_positive_rate_pct'] < 20:
            create_callout(
                f"False positive rate: {result.metrics['false_positive_rate_pct']:.1f}%\n\n"
                f"Only {result.metrics['alerts_during_exercise']} false alarms during exercise.\n\n"
                "Wood Wide successfully distinguishes between normal exercise patterns and genuine "
                "anomalies by understanding signal relationships in embedding space.",
                type="success",
                title="Context-Aware Detection Achieved"
            )

        st.divider()

        # Visualization
        st.subheader("Detection Visualization")

        time_minutes = (timestamps - timestamps[0]) / 60

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Distance from Normal Centroid", "Alert Timeline"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # Distance plot
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=result.distances,
                mode='lines',
                name='Distance',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )

        fig.add_hline(
            y=result.threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {result.threshold:.3f}",
            row=1, col=1
        )

        if result.alerts.any():
            fig.add_trace(
                go.Scatter(
                    x=time_minutes[result.alerts],
                    y=result.distances[result.alerts],
                    mode='markers',
                    name='Alerts',
                    marker=dict(size=12, color='red', symbol='x', line=dict(width=2))
                ),
                row=1, col=1
            )

        # Alert timeline
        fig.add_trace(
            go.Scatter(
                x=time_minutes,
                y=result.alerts.astype(float),
                mode='lines',
                name='Alert Active',
                fill='tozeroy',
                line=dict(color='green', width=0),
                fillcolor='rgba(0, 255, 0, 0.3)'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="Distance", row=1, col=1)
        fig.update_yaxes(title_text="Alert", row=2, col=1, tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])

        fig.update_layout(height=600, showlegend=True, hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Heart Rate vs Accelerometer with Anomaly Shading
        st.subheader("Heart Rate vs. Physical Activity")

        st.markdown("""
        This chart shows the relationship between heart rate and accelerometer magnitude.
        **Red shaded areas** indicate Wood Wide-detected anomalies where the signals are decoupled.
        """)

        # Extract HR for this view
        hr_for_chart = extract_heart_rate_simple(windows)

        # Create HR vs ACC chart
        hr_acc_fig = create_hr_vs_acceleration_chart(
            windows=windows,
            timestamps=timestamps,
            hr_bpm=hr_for_chart,
            woodwide_alerts=result.alerts,
            title="Heart Rate vs. Accelerometer Magnitude (Red = Anomaly)"
        )

        st.plotly_chart(hr_acc_fig, use_container_width=True)

        st.markdown("""
        <div class="info-callout">
        <strong>How to interpret</strong><br><br>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#e74c3c;margin-right:6px;vertical-align:middle;"></span> <strong>Red line</strong> &mdash; Heart rate (left axis)<br>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3498db;margin-right:6px;vertical-align:middle;"></span> <strong>Blue line</strong> &mdash; Accelerometer magnitude (right axis)<br>
        <span style="display:inline-block;width:10px;height:3px;background:#e74c3c;opacity:0.5;margin-right:6px;vertical-align:middle;"></span> <strong>Red shading</strong> &mdash; Wood Wide detected anomaly (signals decoupled)<br><br>
        <strong>Normal:</strong> High HR + High ACC = No shading (exercise is normal)<br>
        <strong>Anomaly:</strong> High HR + Low ACC = Red shading (potential health concern)
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # By activity
        st.subheader("Performance by Activity")

        activity_perf = []
        for label, activity in activity_map.items():
            mask = labels == label
            if mask.sum() > 0:
                n_alerts = (result.alerts & mask).sum()
                mean_dist = result.distances[mask].mean()
                activity_perf.append({
                    'Activity': activity,
                    'Type': 'Exercise' if label in [2, 3, 4, 7] else 'Rest',
                    'Windows': mask.sum(),
                    'Alerts': n_alerts,
                    'Alert Rate (%)': n_alerts / mask.sum() * 100,
                    'Mean Distance': f"{mean_dist:.3f}"
                })

        df_perf = pd.DataFrame(activity_perf)
        st.dataframe(df_perf, hide_index=True, use_container_width=True)

        st.divider()

        # Advanced topics in expandable sections
        with st.expander("How Wood Wide Detection Works"):
            st.markdown(ALGORITHM_EXPLANATIONS["woodwide_how_it_works"])

        with st.expander("Production Deployment Considerations"):
            st.markdown(DEPLOYMENT_CONSIDERATIONS)
            st.markdown("")
            display_code_snippet(
                CODE_SNIPPETS["save_load_detector"],
                caption="Save and Load Trained Detector"
            )

        with st.expander("API Reference"):
            st.markdown(FURTHER_READING["documentation"])
            st.markdown("")
            st.markdown(FURTHER_READING["examples"])

    # Tab 4: Three-Way Comparison
    with tab4:
        st.header("Three-Way Performance Comparison")
        st.markdown("### Comparing Threshold, Isolation Forest, and Wood Wide Detection")

        # Evaluation Methodology
        st.subheader("Evaluation Methodology")
        duration = (timestamps[-1] - timestamps[0]) / 60
        is_exercise = np.isin(labels, [2, 3, 4, 7])

        st.markdown(f"""
        **Detection Methods:**
        1. **Baseline Threshold**: Simple heart rate threshold (HR > {baseline_threshold} BPM)
        2. **Isolation Forest**: Traditional anomaly detection on hand-crafted features
        3. **Wood Wide**: Context-aware detection using multivariate embeddings

        **Evaluation Metrics:**
        - **False Positive Rate:** Alerts during exercise (should be low)
        - **True Positive Rate:** Alerts during rest anomalies (should be high)
        - **Total Alert Count:** System usability indicator

        **Test Dataset:**
        - Subject: {subject_id}
        - Windows: {len(windows)}
        - Duration: {duration:.1f} minutes
        - Exercise periods: {is_exercise.sum()} windows
        - Rest periods: {(~is_exercise).sum()} windows
        """)

        st.divider()

        # Compute all three methods
        hr_bpm = extract_heart_rate_simple(windows)
        baseline_alerts = hr_bpm > baseline_threshold

        # Load Wood Wide results (session state first, then disk cache)
        embeddings_data = st.session_state.embeddings_cache.get(subject_id) or load_embeddings(subject_id)
        if embeddings_data is None:
            st.warning("Please generate embeddings in the Wood Wide tab first.")
            return

        embeddings, _ = embeddings_data
        detector = WoodWideDetector(threshold_percentile=woodwide_percentile)
        result = detector.fit_predict(embeddings, labels)

        # Run Isolation Forest detection
        with st.spinner("Running Isolation Forest detection..."):
            iforest_detector = IsolationForestDetector(contamination=0.1)
            iforest_detector.fit(windows, labels)
            iforest_result = iforest_detector.predict(windows)

            # Compute metrics from the alerts
            iforest_metrics = compute_isolation_forest_metrics(
                if_alerts=iforest_result.alerts,
                labels=labels
            )
            # Add alerts to metrics for timeline visualization
            iforest_metrics['alerts'] = iforest_result.alerts

        # Compute baseline metrics
        is_exercise = np.isin(labels, [2, 3, 4, 7])
        baseline_metrics = {
            'total_alerts': int(baseline_alerts.sum()),
            'alerts_during_exercise': int((baseline_alerts & is_exercise).sum()),
            'alerts_during_rest': int((baseline_alerts & ~is_exercise).sum()),
            'false_positive_rate_pct': (baseline_alerts & is_exercise).sum() / is_exercise.sum() * 100 if is_exercise.sum() > 0 else 0,
            'exercise_windows': int(is_exercise.sum()),
            'rest_windows': int((~is_exercise).sum())
        }

        # Three-way comparison chart
        st.subheader("Performance Comparison")

        three_way_fig = create_three_way_comparison_chart(
            baseline_fp_rate=baseline_metrics['false_positive_rate_pct'],
            if_fp_rate=iforest_metrics['false_positive_rate_pct'],
            woodwide_fp_rate=result.metrics['false_positive_rate_pct']
        )
        st.plotly_chart(three_way_fig, use_container_width=True)

        st.divider()

        # Detailed comparison table
        st.subheader("Detailed Metrics")

        comparison_table = create_comparison_table(
            baseline_metrics=baseline_metrics,
            if_metrics=iforest_metrics,
            woodwide_metrics=result.metrics
        )
        st.dataframe(comparison_table, hide_index=True, use_container_width=True)

        # Key insights
        baseline_fp = baseline_metrics['false_positive_rate_pct']
        iforest_fp = iforest_metrics['false_positive_rate_pct']
        woodwide_fp = result.metrics['false_positive_rate_pct']

        if woodwide_fp < iforest_fp and woodwide_fp < baseline_fp:
            improvement_vs_baseline = baseline_fp - woodwide_fp
            improvement_vs_iforest = iforest_fp - woodwide_fp
            create_callout(
                f"**Wood Wide achieves the lowest false positive rate:**\n\n"
                f"- **{improvement_vs_baseline:.1f}%** better than Threshold Detection\n"
                f"- **{improvement_vs_iforest:.1f}%** better than Isolation Forest\n\n"
                "Wood Wide's context-aware embeddings successfully distinguish exercise from health concerns, "
                "outperforming both traditional threshold methods and hand-crafted feature approaches.",
                type="success",
                title="Best Performance: Wood Wide"
            )

        st.divider()

        # Three-way timeline
        st.subheader("Detection Timeline Comparison")

        timeline_fig = create_three_way_timeline(
            timestamps=timestamps,
            hr_bpm=hr_bpm,
            baseline_alerts=baseline_alerts,
            if_alerts=iforest_metrics['alerts'],
            woodwide_alerts=result.alerts
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        st.markdown("""
        **How to interpret:**
        - **Top panel**: Heart rate over time
        - **Bottom panels**: Alert timelines for each method (red = alert triggered)
        - Notice how Baseline and Isolation Forest trigger many alerts during exercise periods
        - Wood Wide maintains low false positives while still detecting genuine anomalies
        """)

        st.divider()

        # Signal Relationship Visualization
        st.subheader("Signal Relationship Analysis")

        st.markdown("""
        This dual-axis chart reveals **why Wood Wide succeeds where threshold detection fails**.
        Watch how the red anomaly regions correspond to signal decoupling.
        """)

        # Create HR vs ACC chart with anomalies
        hr_acc_comparison = create_hr_vs_acceleration_chart(
            windows=windows,
            timestamps=timestamps,
            hr_bpm=hr_bpm,
            woodwide_alerts=result.alerts,
            title="The Context Problem Visualized: Heart Rate vs. Activity"
        )

        st.plotly_chart(hr_acc_comparison, use_container_width=True)

        # Interpretation guide
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="success-callout">
            <strong>Normal Pattern</strong><br>
            High HR + High ACC = No anomaly<br>
            <em>Example: Exercising (cycling, walking)</em><br>
            Wood Wide understands this is normal.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="warning-callout">
            <strong>Anomaly Pattern</strong><br>
            High HR + Low ACC = Anomaly (red shading)<br>
            <em>Example: Elevated HR while sitting</em><br>
            Wood Wide detects signal decoupling.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Reproducing Results
        st.subheader("Reproducing Results")
        display_code_snippet(
            CODE_SNIPPETS["comparison_workflow"],
            caption="Complete Comparison Workflow"
        )

        st.divider()

        # Conclusion
        st.subheader("Conclusion")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Threshold Detection")
            st.markdown("""**Limitations:**
1. **Context-blind**: Examines HR in isolation
2. **High false positive rate**: 80-100% during exercise
3. **Unusable for continuous monitoring**: Alert fatigue
4. **No learning**: Cannot adapt to patterns""")

        with col2:
            st.markdown("### Isolation Forest")
            st.markdown("""**Moderate Performance:**
1. **Hand-crafted features**: Requires domain expertise
2. **Improved over threshold**: Better than baseline
3. **Still context-limited**: Features don't capture relationships
4. **Configuration-sensitive**: Contamination parameter critical""")

        with col3:
            st.markdown("### Wood Wide (Best)")
            st.markdown("""**Advantages:**
1. **Context-aware**: Understands HR-activity relationship
2. **Low false positive rate**: <10% during exercise
3. **Practical for deployment**: Manageable alert rate
4. **Embedding-based**: Learns relationships automatically""")

        st.markdown("""
        ### The Context Problem: Solved

        Multivariate embeddings enable context-aware detection by encoding signal relationships in latent space.
        This fundamental capability cannot be achieved with threshold-based methods **or** hand-crafted features,
        regardless of optimization.

        **Wood Wide doesn't just set a better threshold or engineer better features—it solves a fundamentally
        different problem by understanding how signals relate to each other in a learned representation space.**
        """)

        # Further Reading
        st.subheader("Further Reading")
        st.markdown(FURTHER_READING["documentation"])
        st.markdown(FURTHER_READING["examples"])
        st.markdown(FURTHER_READING["research"])


if __name__ == "__main__":
    main()
