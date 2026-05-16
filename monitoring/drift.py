"""
monitoring/drift.py
────────────────────
Lightweight drift and spike detection for forecast monitoring.
No external monitoring tools — pure numpy/pandas.
"""

import numpy as np


def rolling_wmape(
    actual: np.ndarray,
    pred:   np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling WMAPE over a sliding window.

    Returns array of length (len(actual) - window + 1).
    Each value is the WMAPE for a window-sized chunk.
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    n      = len(actual)

    if n < window:
        # Return single value if series is too short
        denom = np.sum(np.abs(actual))
        return np.array([np.sum(np.abs(actual - pred)) / denom * 100 if denom > 0 else np.nan])

    result = []
    for i in range(n - window + 1):
        a_w = actual[i : i + window]
        p_w = pred  [i : i + window]
        d   = np.sum(np.abs(a_w))
        if d > 0:
            result.append(np.sum(np.abs(a_w - p_w)) / d * 100)
        else:
            result.append(np.nan)

    return np.array(result)


def rolling_mape(
    actual: np.ndarray,
    pred:   np.ndarray,
    window: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute rolling MAPE over a sliding window."""
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    n      = len(actual)

    if n < window:
        mask = actual > epsilon
        if mask.sum() == 0:
            return np.array([np.nan])
        return np.array([np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100])

    result = []
    for i in range(n - window + 1):
        a_w  = actual[i : i + window]
        p_w  = pred  [i : i + window]
        mask = a_w > epsilon
        if mask.sum() > 0:
            result.append(np.mean(np.abs(a_w[mask] - p_w[mask]) / a_w[mask]) * 100)
        else:
            result.append(np.nan)

    return np.array(result)


def detect_drift(
    actual:    np.ndarray,
    pred:      np.ndarray,
    threshold: float = 1.3,
) -> dict:
    """
    Compare WMAPE in the first half vs second half of the forecast period.
    A ratio > threshold indicates model degradation / drift.

    Parameters
    ----------
    actual    : float array [horizon]
    pred      : float array [horizon]
    threshold : ratio of late_wmape / early_wmape to flag as drift

    Returns
    -------
    dict with keys: drift_detected, early_wmape, late_wmape, ratio, message
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    n      = len(actual)
    mid    = max(n // 2, 1)

    a_early, p_early = actual[:mid], pred[:mid]
    a_late,  p_late  = actual[mid:], pred[mid:]

    def _wmape(a, p):
        d = np.sum(np.abs(a))
        return float(np.sum(np.abs(a - p)) / d * 100) if d > 0 else np.nan

    early_w = _wmape(a_early, p_early)
    late_w  = _wmape(a_late,  p_late)

    if np.isnan(early_w) or early_w == 0:
        ratio   = 1.0
        drifted = False
    else:
        ratio   = late_w / early_w if not np.isnan(late_w) else 1.0
        drifted = ratio > threshold

    return {
        "drift_detected": drifted,
        "early_wmape":    round(early_w, 2) if not np.isnan(early_w) else 0.0,
        "late_wmape":     round(late_w,  2) if not np.isnan(late_w)  else 0.0,
        "ratio":          round(ratio, 3),
        "threshold":      threshold,
        "message": (
            f"⚠️ Drift detected — WMAPE increased {round(early_w,1)}% → {round(late_w,1)}% "
            f"(ratio {round(ratio,2)}×). Consider retraining."
            if drifted else
            f"✅ Model stable — WMAPE {round(early_w,1)}% → {round(late_w,1)}%"
        ),
    }


def detect_spikes(
    errors:      np.ndarray,
    z_threshold: float = 2.5,
) -> np.ndarray:
    """
    Flag forecast steps where absolute error is unusually high.

    A step is flagged if its error > mean + z_threshold × std.

    Returns
    -------
    bool array of same length as errors — True = spike detected
    """
    errors = np.asarray(errors, dtype=float)
    if len(errors) == 0:
        return np.array([], dtype=bool)

    mean_err = np.nanmean(errors)
    std_err  = np.nanstd(errors)

    if std_err == 0:
        return np.zeros(len(errors), dtype=bool)

    return errors > (mean_err + z_threshold * std_err)


def forecast_stability_score(
    rolling_wmape_vals: np.ndarray,
) -> float:
    """
    Score between 0–100 measuring how stable the rolling WMAPE is.
    100 = perfectly flat WMAPE; 0 = highly volatile.
    Based on coefficient of variation (lower CV = higher score).
    """
    vals = np.asarray(rolling_wmape_vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0 or np.mean(vals) == 0:
        return 100.0

    cv = np.std(vals) / np.mean(vals)
    score = max(0.0, 100.0 - cv * 100.0)
    return round(score, 1)
