"""
visualizations/monitoring_plots.py
────────────────────────────────────
Monitoring-specific charts: drift, spikes, stability.
"""

import numpy as np
import plotly.graph_objects as go

from visualizations.plots import DARK_LAYOUT, COLORS


def plot_drift_visualization(
    timestamps,
    actual:    np.ndarray,
    tfm_preds: np.ndarray,
    chr_preds: np.ndarray,
    drift_info: dict,
) -> go.Figure:
    """
    Actual vs forecast chart with drift region highlighted.
    The second half of the forecast window is shaded red if drift is detected.
    """
    timestamps = list(timestamps)
    n          = len(actual)
    mid        = n // 2

    fig = go.Figure()

    # Drift shading on second half
    if drift_info.get("drift_detected") and len(timestamps) > mid:
        fig.add_vrect(
            x0=timestamps[mid], x1=timestamps[-1],
            fillcolor=COLORS["drift_bg"],
            layer="below",
            line_width=0,
            annotation_text="Drift region",
            annotation_position="top left",
            annotation_font_color="#f85149",
        )

    # Actual
    fig.add_trace(go.Scatter(
        x=timestamps, y=actual,
        name="Actual", mode="lines+markers",
        line=dict(color=COLORS["actual"], width=2),
        marker=dict(size=4),
        hovertemplate="%{x}<br>Actual: %{y:.0f}<extra></extra>",
    ))

    # TimesFM
    fig.add_trace(go.Scatter(
        x=timestamps, y=tfm_preds,
        name="TimesFM", mode="lines",
        line=dict(color=COLORS["timesfm"], width=2, dash="dash"),
        hovertemplate="%{x}<br>TimesFM: %{y:.0f}<extra></extra>",
    ))

    # Chronos
    fig.add_trace(go.Scatter(
        x=timestamps, y=chr_preds,
        name="Chronos", mode="lines",
        line=dict(color=COLORS["chronos"], width=2, dash="dashdot"),
        hovertemplate="%{x}<br>Chronos: %{y:.0f}<extra></extra>",
    ))

    # Midpoint marker
    if len(timestamps) > mid:
        x_val = timestamps[mid]
        if hasattr(x_val, "timestamp"):
            x_val = x_val.timestamp() * 1000
            
        fig.add_vline(
            x=x_val,
            line_dash="dot", line_color="#888", line_width=1,
            annotation_text="Half-way",
            annotation_position="top",
            annotation_font_color="#888",
        )

    status = "⚠️ DRIFT" if drift_info.get("drift_detected") else "✅ STABLE"
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Drift Analysis — {status}",
        height=340,
        xaxis=dict(title="Step", showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="Order Qty", showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.3),
        hovermode="x unified",
    )
    return fig


def plot_spike_overlay(
    timestamps,
    errors:      np.ndarray,
    spike_flags: np.ndarray,
    model_name:  str = "TimesFM",
    color:       str = None,
) -> go.Figure:
    """
    Error line chart with spike positions marked as red dots.
    """
    timestamps = list(timestamps)
    color      = color or COLORS["timesfm"]

    fig = go.Figure()

    # Error line
    fig.add_trace(go.Scatter(
        x=timestamps, y=errors,
        name=f"{model_name} Error",
        mode="lines",
        line=dict(color=color, width=1.8),
        hovertemplate="%{x}<br>Error: %{y:.1f}<extra></extra>",
    ))

    # Spike markers
    spike_idx = np.where(spike_flags)[0]
    if len(spike_idx) > 0:
        spike_ts  = [timestamps[i] for i in spike_idx if i < len(timestamps)]
        spike_err = errors[spike_idx[spike_idx < len(errors)]]
        fig.add_trace(go.Scatter(
            x=spike_ts, y=spike_err,
            name="Spikes",
            mode="markers",
            marker=dict(color=COLORS["spike"], size=10, symbol="x"),
            hovertemplate="%{x}<br>SPIKE Error: %{y:.1f}<extra>Spike</extra>",
        ))

    # Threshold line
    mean_err = np.nanmean(errors)
    std_err  = np.nanstd(errors)
    threshold = mean_err + 2.5 * std_err
    fig.add_hline(
        y=threshold,
        line_dash="dot", line_color="#f85149", line_width=1,
        annotation_text=f"Spike threshold ({threshold:.1f})",
        annotation_position="bottom right",
        annotation_font_color="#f85149",
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Spike Detection — {model_name}",
        height=300,
        xaxis=dict(title="Step", showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="Absolute Error", showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.3),
        hovermode="x unified",
    )
    return fig


def plot_stability_gauge(score: float, model_name: str = "Model") -> go.Figure:
    """
    Gauge chart for forecast stability score (0–100).
    100 = perfectly stable, 0 = highly volatile.
    """
    color = (
        "#7ee787" if score >= 70  else
        "#f0c030" if score >= 40  else
        "#f85149"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(text=f"{model_name} Stability Score", font=dict(color="#c9d1d9")),
        number=dict(suffix="/100", font=dict(color=color, size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#c9d1d9"),
            bar=dict(color=color),
            bgcolor="#1a1d27",
            borderwidth=1,
            bordercolor="#2e3347",
            steps=[
                dict(range=[0,   40], color="#2d1a1a"),
                dict(range=[40,  70], color="#2a2515"),
                dict(range=[70, 100], color="#1a2d1a"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=70,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        font=dict(color="#c9d1d9"),
        height=240,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig
