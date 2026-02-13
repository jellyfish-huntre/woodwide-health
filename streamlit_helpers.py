"""
Helper functions for Streamlit dashboard three-way comparison.

Provides utilities for comparing Baseline Threshold, Isolation Forest,
and Wood Wide detection methods.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def compute_isolation_forest_metrics(if_alerts: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute performance metrics for Isolation Forest detection."""
    is_exercise = np.isin(labels, [2, 3, 4, 5])

    return {
        'total_alerts': int(if_alerts.sum()),
        'alerts_during_exercise': int((if_alerts & is_exercise).sum()),
        'alerts_during_rest': int((if_alerts & ~is_exercise).sum()),
        'false_positive_rate_pct': (
            (if_alerts & is_exercise).sum() / is_exercise.sum() * 100
            if is_exercise.sum() > 0 else 0
        ),
        'exercise_windows': int(is_exercise.sum()),
        'rest_windows': int((~is_exercise).sum())
    }


def create_three_way_comparison_chart(
    baseline_fp_rate: float,
    if_fp_rate: float,
    woodwide_fp_rate: float
) -> go.Figure:
    """
    Create bar chart comparing false positive rates across three methods.

    Args:
        baseline_fp_rate: Baseline threshold FP rate (%)
        if_fp_rate: Isolation Forest FP rate (%)
        woodwide_fp_rate: Wood Wide FP rate (%)

    Returns:
        Plotly figure
    """
    methods = ['Naive<br>Threshold', 'Isolation<br>Forest', 'Wood Wide']
    fp_rates = [baseline_fp_rate, if_fp_rate, woodwide_fp_rate]
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=methods,
        y=fp_rates,
        marker_color=colors,
        text=[f'{rate:.1f}%' for rate in fp_rates],
        textposition='outside',
        textfont=dict(size=14, family='Inter'),
        hovertemplate='%{x}<br>FP Rate: %{y:.1f}%<extra></extra>'
    ))

    max_rate = max(fp_rates)
    fig.update_layout(
        title=dict(
            text="Exercise False Positive Rates",
            font=dict(size=16, family='Inter')
        ),
        yaxis=dict(
            title="False Positive Rate (%)",
            range=[0, max(max_rate * 1.15, max_rate + 10)]
        ),
        xaxis_title="Detection Method",
        height=450,
        showlegend=False,
    )

    return fig


def create_three_way_timeline(
    timestamps: np.ndarray,
    hr_bpm: np.ndarray,
    baseline_alerts: np.ndarray,
    if_alerts: np.ndarray,
    woodwide_alerts: np.ndarray
) -> go.Figure:
    """
    Create timeline showing all three detection methods' alerts.

    Args:
        timestamps: Window timestamps
        hr_bpm: Heart rate in BPM
        baseline_alerts: Boolean array for baseline
        if_alerts: Boolean array for Isolation Forest
        woodwide_alerts: Boolean array for Wood Wide

    Returns:
        Plotly figure with three subplot rows
    """
    time_minutes = (timestamps - timestamps[0]) / 60

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Heart Rate',
            'Baseline Threshold Alerts',
            'Isolation Forest Alerts',
            'Wood Wide Alerts'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # Row 1: Heart Rate
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=hr_bpm,
            mode='lines',
            name='Heart Rate',
            line=dict(color='#3498db', width=2),
            hovertemplate='HR: %{y:.1f} BPM<extra></extra>'
        ),
        row=1, col=1
    )

    # Row 2: Baseline alerts
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=baseline_alerts.astype(int),
            mode='lines',
            name='Baseline',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            hovertemplate='Alert: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # Row 3: Isolation Forest alerts
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=if_alerts.astype(int),
            mode='lines',
            name='Isolation Forest',
            line=dict(color='#f39c12', width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.3)',
            hovertemplate='Alert: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Row 4: Wood Wide alerts
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=woodwide_alerts.astype(int),
            mode='lines',
            name='Wood Wide',
            line=dict(color='#27ae60', width=2),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.3)',
            hovertemplate='Alert: %{y}<extra></extra>'
        ),
        row=4, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="BPM", row=1, col=1)

    for row in [2, 3, 4]:
        fig.update_yaxes(
            title_text="Alert",
            row=row, col=1,
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['No', 'Yes'],
            range=[-0.1, 1.1]
        )

    fig.update_layout(
        height=700,
        showlegend=False,
        hovermode='x unified',
    )

    return fig


def create_comparison_table(
    baseline_metrics: Dict,
    if_metrics: Dict,
    woodwide_metrics: Dict
) -> pd.DataFrame:
    """
    Create comparison table for all three methods.

    Args:
        baseline_metrics: Metrics for baseline threshold
        if_metrics: Metrics for Isolation Forest
        woodwide_metrics: Metrics for Wood Wide

    Returns:
        DataFrame with comparison data
    """
    return pd.DataFrame({
        'Metric': [
            'Total Alerts',
            'Exercise Alerts (FP)',
            'Rest Alerts',
            'Exercise FP Rate',
            'Exercise Windows',
            'Rest Windows'
        ],
        'Baseline': [
            baseline_metrics['total_alerts'],
            baseline_metrics['alerts_during_exercise'],
            baseline_metrics['alerts_during_rest'],
            f"{baseline_metrics['false_positive_rate_pct']:.1f}%",
            baseline_metrics['exercise_windows'],
            baseline_metrics['rest_windows']
        ],
        'Isolation Forest': [
            if_metrics['total_alerts'],
            if_metrics['alerts_during_exercise'],
            if_metrics['alerts_during_rest'],
            f"{if_metrics['false_positive_rate_pct']:.1f}%",
            if_metrics['exercise_windows'],
            if_metrics['rest_windows']
        ],
        'Wood Wide': [
            woodwide_metrics['total_alerts'],
            woodwide_metrics['alerts_during_exercise'],
            woodwide_metrics['alerts_during_rest'],
            f"{woodwide_metrics['false_positive_rate_pct']:.1f}%",
            woodwide_metrics['exercise_windows'],
            woodwide_metrics['rest_windows']
        ]
    })
