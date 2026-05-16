"""
visualizations/comparison_plots.py
────────────────────────────────────
Model comparison and leaderboard charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from visualizations.plots import DARK_LAYOUT, COLORS


def plot_metrics_bar(
    per_ser_df: pd.DataFrame,
    metric_key: str = "WMAPE_%",
    title: str = None,
) -> go.Figure:
    """
    Side-by-side bar chart of the chosen metric per series for both models.
    Bars are color-coded: green (good) → red (poor).
    """
    if per_ser_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    series  = sorted(per_ser_df["series_id"].unique())
    x       = np.arange(len(series))
    width   = 0.35
    label   = metric_key.replace("_", " ").replace("%", " %")

    fig = go.Figure()

    for i, (model, color) in enumerate([("timesfm", COLORS["timesfm"]), ("chronos", COLORS["chronos"])]):
        vals = []
        for s in series:
            row = per_ser_df[(per_ser_df["series_id"] == s) & (per_ser_df["model"] == model)]
            vals.append(float(row[metric_key].values[0]) if not row.empty else np.nan)

        fig.add_trace(go.Bar(
            name=model.upper(),
            x=series,
            y=vals,
            marker_color=color,
            opacity=0.85,
            text=[f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in vals],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{label}: %{{y:.1f}}%<extra>{model.upper()}</extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=title or f"{label} per Series — TimesFM vs Chronos",
        height=380,
        barmode="group",
        xaxis=dict(title="Series", showgrid=False),
        yaxis=dict(title=label, showgrid=True, gridcolor="#2e3347"),
        legend=dict(orientation="h", y=-0.25, x=0.0),
    )
    return fig


def plot_error_distribution(results_df: pd.DataFrame) -> go.Figure:
    """
    Violin chart comparing the distribution of absolute % errors
    for TimesFM vs Chronos across all series.
    """
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    actual = results_df["actual"].values
    eps    = 1e-8

    fig = go.Figure()

    for model_col, model_name, color in [
        ("timesfm_pred", "TimesFM", COLORS["timesfm"]),
        ("chronos_pred",  "Chronos", COLORS["chronos"]),
    ]:
        if model_col not in results_df.columns:
            continue
        preds  = results_df[model_col].values
        mask   = np.abs(actual) > eps
        errors = np.abs(actual[mask] - preds[mask]) / np.abs(actual[mask]) * 100

        fig.add_trace(go.Violin(
            y=errors,
            name=model_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color.replace(")", ", 0.3)").replace("rgb", "rgba"),
            line_color=color,
            opacity=0.8,
            hovertemplate=f"Error %%: %{{y:.1f}}<extra>{model_name}</extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="Forecast Error Distribution (% error per step)",
        height=350,
        yaxis=dict(title="Absolute % Error", showgrid=True, gridcolor="#2e3347"),
        xaxis=dict(title="Model"),
        violinmode="group",
    )
    return fig


def plot_series_ranking(
    per_ser_df: pd.DataFrame,
    metric_key: str = "WMAPE_%",
) -> go.Figure:
    """
    Horizontal bar chart ranking series from best to worst by metric.
    Color: green → yellow → red by metric value.
    """
    if per_ser_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    # Pivot: best model per series
    pivot = per_ser_df.groupby("series_id")[metric_key].min().reset_index()
    pivot = pivot.sort_values(metric_key, ascending=True)
    label = metric_key.replace("_", " ").replace("%", " %")

    # Color scale: green → red
    vals   = pivot[metric_key].values
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    norm   = (vals - v_min) / (v_max - v_min + 1e-8)
    colors = [
        f"rgb({int(50 + 200 * n)}, {int(200 - 180 * n)}, {int(80 - 60 * n)})"
        for n in norm
    ]

    fig = go.Figure(go.Bar(
        y=pivot["series_id"],
        x=pivot[metric_key],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate=f"%{{y}}<br>Best {label}: %{{x:.1f}}%<extra></extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Series Ranking by Best {label} (lower = better)",
        height=max(300, 60 * len(pivot)),
        xaxis=dict(title=label, showgrid=True, gridcolor="#2e3347"),
        yaxis=dict(title="Series", automargin=True),
    )
    return fig


def plot_leaderboard(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly table showing full metrics leaderboard.
    """
    if metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    cols = ["series_id", "model", "WMAPE_%", "MAPE_%"]
    if "MAE" in metrics_df.columns:
        cols += ["MAE", "RMSE"]
    cols += ["n_steps"]
    df = metrics_df[cols].copy()

    # Color WMAPE column: low=green, high=red
    wmape_vals = df["WMAPE_%"].values
    v_min, v_max = np.nanmin(wmape_vals), np.nanmax(wmape_vals)
    norm   = (wmape_vals - v_min) / (v_max - v_min + 1e-8)
    cell_colors = [
        [f"rgba({int(50 + 200 * n)}, {int(200 - 180 * n)}, {int(80 - 60 * n)}, 0.25)"
         for n in norm]
    ]

    header_vals = [c.replace("_", " ") for c in cols]
    cell_vals   = [df[c].tolist() for c in cols]

    fill_colors = ["#1e2130"] * len(cols)

    fig = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color="#2e3347",
            align="left",
            font=dict(color="white", size=12),
            line_color="#3d4466",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=fill_colors,
            align="left",
            font=dict(color="#c9d1d9", size=11),
            line_color="#2e3347",
            height=28,
        ),
    ))
    # Merge margins with DARK_LAYOUT
    layout_args = DARK_LAYOUT.copy()
    layout_args["title"] = "Full Metrics Leaderboard"
    layout_args["height"] = max(350, 40 * len(df))
    layout_args["margin"] = dict(t=50, b=20, l=10, r=10)
    
    fig.update_layout(**layout_args)
    return fig


def plot_model_winner_pie(
    per_ser_df: pd.DataFrame,
    metric_key: str = "WMAPE_%",
) -> go.Figure:
    """
    Pie chart showing how many series each model wins on the given metric.
    """
    if per_ser_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    wins = {"TimesFM": 0, "Chronos": 0, "Tie": 0}
    for sid in per_ser_df["series_id"].unique():
        grp = per_ser_df[per_ser_df["series_id"] == sid]
        t   = grp[grp["model"] == "timesfm"][metric_key].values
        c   = grp[grp["model"] == "chronos"][metric_key].values
        if len(t) == 0 or len(c) == 0:
            continue
        if t[0] < c[0]:
            wins["TimesFM"] += 1
        elif c[0] < t[0]:
            wins["Chronos"] += 1
        else:
            wins["Tie"] += 1

    labels = [k for k, v in wins.items() if v > 0]
    values = [v for v in wins.values() if v > 0]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=[COLORS["timesfm"], COLORS["chronos"], "#888"]),
        textinfo="label+percent",
        hole=0.4,
        hovertemplate="%{label}: %{value} series<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Model Wins by {metric_key.replace('_', ' ')}",
        height=320,
        showlegend=False,
    )
    return fig
