"""
visualizations/plots.py
────────────────────────
Core forecast visualization charts using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ── Shared theme ──────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0f1117",
    plot_bgcolor="#1a1d27",
    font=dict(family="Inter, sans-serif", color="#c9d1d9"),
    margin=dict(t=50, b=40, l=50, r=20),
)

COLORS = {
    "history":  "#5c6370",
    "actual":   "#58a6ff",
    "timesfm":  "#ff7b54",
    "chronos":  "#7ee787",
    "ci_tfm":   "rgba(255, 123, 84, 0.12)",
    "ci_chr":   "rgba(126, 231, 135, 0.12)",
    "spike":    "#f85149",
    "drift_bg": "rgba(248, 81, 73, 0.08)",
}


def plot_historical_overview(df, series_ids: list = None, n_hist: int = 90) -> go.Figure:
    """
    Line chart of recent order quantities across all (or selected) series.
    """
    import pandas as pd

    if series_ids:
        df = df[df["series_id"].isin(series_ids)]

    hist_df = df.groupby("series_id").tail(n_hist).reset_index(drop=True)

    fig = px.line(
        hist_df,
        x="timestamp",
        y="order_qty",
        color="series_id",
        labels={"order_qty": "Order Qty", "timestamp": "Date", "series_id": "Series"},
        title=f"Historical Order Quantities — Last {n_hist} Days",
    )
    fig.update_traces(line=dict(width=1.8))
    fig.update_layout(
        **DARK_LAYOUT,
        height=350,
        legend=dict(orientation="h", y=-0.25, x=0.0),
        xaxis=dict(showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(showgrid=True, gridcolor="#2e3347"),
    )
    return fig


def plot_forecast_interactive(
    series_df,
    results:    dict,
    sid:        str,
    n_hist:     int  = 30,
    show_tfm:   bool = True,
    show_chr:   bool = True,
    show_ci:    bool = True,
) -> go.Figure:
    """
    Full interactive forecast chart: history + actual + TimesFM + Chronos + CI bands.

    Parameters
    ----------
    series_df : full series DataFrame (for history)
    results   : dict from run_single_series()
    sid       : series label
    n_hist    : number of historical points to display
    show_tfm  : show TimesFM trace
    show_chr  : show Chronos trace
    show_ci   : show confidence interval bands
    """
    import pandas as pd

    actual     = np.array(results["actual"])
    timestamps = results["timestamp"]
    n_fore     = len(actual)

    # History (everything before the test window)
    hist = series_df[series_df["series_id"] == sid]["order_qty"].values if "series_id" in series_df.columns else series_df["order_qty"].values
    hist_window = hist[-(n_hist + n_fore): -n_fore] if len(hist) > n_fore else hist
    hist_ts_idx = list(range(-len(hist_window), 0))
    fore_ts_idx = list(range(0, n_fore))

    # Use actual timestamps if available
    if len(timestamps) == n_fore:
        x_fore = list(timestamps)
        # Get historical timestamps from the actual series DataFrame
        if "timestamp" in series_df.columns:
            all_ts = series_df["timestamp"].values
            hist_start = max(0, len(all_ts) - n_fore - len(hist_window))
            hist_end   = max(0, len(all_ts) - n_fore)
            x_hist = list(all_ts[hist_start:hist_end])
        else:
            # Fallback: estimate using timedelta
            fore_pd  = pd.to_datetime(x_fore)
            if n_fore > 1:
                freq_est = (fore_pd[-1] - fore_pd[0]) / (n_fore - 1)
            else:
                freq_est = pd.Timedelta(days=1)
            x_hist = [fore_pd[0] - freq_est * (len(hist_window) - i) for i in range(len(hist_window))]
    else:
        x_hist = hist_ts_idx
        x_fore = fore_ts_idx

    fig = go.Figure()

    # History
    fig.add_trace(go.Scatter(
        x=x_hist, y=hist_window,
        name="History", mode="lines",
        line=dict(color=COLORS["history"], width=1.5, dash="dot"),
        hovertemplate="%{x}<br>Qty: %{y:.0f}<extra>History</extra>",
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=x_fore, y=actual,
        name="Actual", mode="lines+markers",
        line=dict(color=COLORS["actual"], width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x}<br>Actual: %{y:.0f}<extra>Actual</extra>",
    ))

    # TimesFM
    if show_tfm and "timesfm_pred" in results:
        tfm  = np.array(results["timesfm_pred"])
        fig.add_trace(go.Scatter(
            x=x_fore, y=tfm,
            name="TimesFM", mode="lines",
            line=dict(color=COLORS["timesfm"], width=2, dash="dash"),
            hovertemplate="%{x}<br>TimesFM: %{y:.0f}<extra>TimesFM</extra>",
        ))
        if show_ci and "timesfm_lower" in results:
            lo = np.array(results["timesfm_lower"])
            hi = np.array(results["timesfm_upper"])
            fig.add_trace(go.Scatter(
                x=list(x_fore) + list(x_fore[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=COLORS["ci_tfm"],
                line=dict(color="rgba(0,0,0,0)"),
                name="TimesFM CI",
                showlegend=True,
                hoverinfo="skip",
            ))

    # Chronos
    if show_chr and "chronos_pred" in results:
        chr_ = np.array(results["chronos_pred"])
        fig.add_trace(go.Scatter(
            x=x_fore, y=chr_,
            name="Chronos", mode="lines",
            line=dict(color=COLORS["chronos"], width=2, dash="dashdot"),
            hovertemplate="%{x}<br>Chronos: %{y:.0f}<extra>Chronos</extra>",
        ))
        if show_ci and "chronos_lower" in results:
            lo = np.array(results["chronos_lower"])
            hi = np.array(results["chronos_upper"])
            fig.add_trace(go.Scatter(
                x=list(x_fore) + list(x_fore[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=COLORS["ci_chr"],
                line=dict(color="rgba(0,0,0,0)"),
                name="Chronos CI",
                showlegend=True,
                hoverinfo="skip",
            ))

    # Forecast start line
    if len(x_fore) > 0:
        x_val = x_fore[0]
        # Convert timestamp to epoch ms float to avoid Plotly add_vline math bugs
        if hasattr(x_val, "timestamp"):
            x_val = x_val.timestamp() * 1000
            
        fig.add_vline(
            x=x_val,
            line_dash="dot", line_color="white", line_width=1,
            annotation_text="Forecast start",
            annotation_position="top right",
            annotation_font_color="white",
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Forecast vs Actual — {sid}",
        height=420,
        xaxis=dict(title="Date / Step", showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="Order Qty",  showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.25, x=0.0),
        hovermode="x unified",
    )
    return fig


def plot_rolling_wmape(
    timestamps,
    rw_tfm: np.ndarray,
    rw_chr: np.ndarray,
    title: str = "Rolling WMAPE",
) -> go.Figure:
    """Rolling WMAPE line chart for both models."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(timestamps), y=rw_tfm,
        name="TimesFM", mode="lines+markers",
        line=dict(color=COLORS["timesfm"], width=2),
        marker=dict(size=4),
        hovertemplate="%{x}<br>WMAPE: %{y:.1f}%<extra>TimesFM</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(timestamps), y=rw_chr,
        name="Chronos", mode="lines+markers",
        line=dict(color=COLORS["chronos"], width=2),
        marker=dict(size=4),
        hovertemplate="%{x}<br>WMAPE: %{y:.1f}%<extra>Chronos</extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=320,
        xaxis=dict(title="Step", showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="WMAPE %", showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.3),
        hovermode="x unified",
    )
    return fig


def plot_error_trend(
    timestamps,
    err_tfm:  np.ndarray,
    err_chr:  np.ndarray = None,
    title:    str = "Absolute Forecast Error",
) -> go.Figure:
    """Absolute error bar/line chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(timestamps), y=err_tfm,
        name="TimesFM Error",
        marker_color=COLORS["timesfm"],
        opacity=0.75,
        hovertemplate="%{x}<br>Error: %{y:.1f}<extra>TimesFM</extra>",
    ))
    if err_chr is not None:
        fig.add_trace(go.Scatter(
            x=list(timestamps), y=err_chr,
            name="Chronos Error", mode="lines+markers",
            line=dict(color=COLORS["chronos"], width=2),
            marker=dict(size=4),
            hovertemplate="%{x}<br>Error: %{y:.1f}<extra>Chronos</extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=300,
        barmode="overlay",
        xaxis=dict(title="Step", showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="Absolute Error", showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.3),
        hovermode="x unified",
    )
    return fig
