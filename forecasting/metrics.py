"""
forecasting/metrics.py
──────────────────────
WMAPE (primary), MAPE, MAE, RMSE computation.
All functions handle edge cases (zeros, NaN, empty arrays) gracefully.
"""

import numpy as np
import pandas as pd


def mape(actual: np.ndarray, pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Formula : mean( |actual - pred| / max(|actual|, ε) ) × 100
    Caution : inflated when actuals are near zero — prefer WMAPE for volumes.
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    mask   = actual > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - pred[mask]) / np.abs(actual[mask])) * 100)


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error  ← PRIMARY METRIC.

    Formula : sum(|actual - pred|) / sum(|actual|) × 100
    Why use : weights errors by volume → robust to near-zero actuals.
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    denom  = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-series and overall WMAPE / MAPE for both models.

    Input  : DataFrame with [series_id, actual, timesfm_pred, chronos_pred]
    Output : DataFrame with [series_id, model, WMAPE_%, MAPE_%, n_steps]
    """
    rows = []
    for sid, grp in results_df.groupby("series_id"):
        a = grp["actual"].values
        for model_col in ["timesfm_pred", "chronos_pred"]:
            if model_col not in grp.columns:
                continue
            p = grp[model_col].values
            rows.append({
                "series_id": sid,
                "model":     model_col.replace("_pred", ""),
                "WMAPE_%":   round(wmape(a, p), 3),
                "MAPE_%":    round(mape(a, p),  3),
                "n_steps":   len(a),
            })

    per_series = pd.DataFrame(rows)

    # Overall micro-average across all series
    overall_rows = []
    for model_col in ["timesfm_pred", "chronos_pred"]:
        if model_col not in results_df.columns:
            continue
        overall_rows.append({
            "series_id": "OVERALL",
            "model":     model_col.replace("_pred", ""),
            "WMAPE_%":   round(wmape(results_df["actual"].values,
                                    results_df[model_col].values), 3),
            "MAPE_%":    round(mape(results_df["actual"].values,
                                   results_df[model_col].values), 3),
            "n_steps":   len(results_df),
        })

    return pd.concat([per_series, pd.DataFrame(overall_rows)], ignore_index=True)


def compute_metrics_extended(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended metrics including MAE and RMSE for the comparison page.
    """
    rows = []
    for sid, grp in results_df.groupby("series_id"):
        a = grp["actual"].values
        for model_col in ["timesfm_pred", "chronos_pred"]:
            if model_col not in grp.columns:
                continue
            p = grp[model_col].values
            rows.append({
                "series_id": sid,
                "model":     model_col.replace("_pred", ""),
                "WMAPE_%":   round(wmape(a, p), 2),
                "MAPE_%":    round(mape(a, p),  2),
                "MAE":       round(mae(a, p),   2),
                "RMSE":      round(rmse(a, p),  2),
                "n_steps":   len(a),
            })

    per_series = pd.DataFrame(rows)

    overall_rows = []
    for model_col in ["timesfm_pred", "chronos_pred"]:
        if model_col not in results_df.columns:
            continue
        a = results_df["actual"].values
        p = results_df[model_col].values
        overall_rows.append({
            "series_id": "OVERALL",
            "model":     model_col.replace("_pred", ""),
            "WMAPE_%":   round(wmape(a, p), 2),
            "MAPE_%":    round(mape(a, p),  2),
            "MAE":       round(mae(a, p),   2),
            "RMSE":      round(rmse(a, p),  2),
            "n_steps":   len(results_df),
        })

    return pd.concat([per_series, pd.DataFrame(overall_rows)], ignore_index=True)
