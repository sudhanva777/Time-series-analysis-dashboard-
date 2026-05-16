"""
monitoring/monitoring_metrics.py
─────────────────────────────────
Compute the full monitoring summary for a single series.
"""

import numpy as np
import pandas as pd

from monitoring.drift  import (
    rolling_wmape,
    rolling_mape,
    detect_drift,
    detect_spikes,
    forecast_stability_score,
)
from monitoring.alerts import generate_alerts
from forecasting.metrics import wmape, mape, mae, rmse


def compute_monitoring_summary(
    actual:     np.ndarray,
    tfm_preds:  np.ndarray,
    chr_preds:  np.ndarray,
    timestamps: list,
    window:     int = 7,
    wmape_threshold: float = 30.0,
    drift_threshold: float = 1.3,
    spike_z:         float = 2.5,
) -> dict:
    """
    Compute the full monitoring bundle for a series.

    Returns
    -------
    dict with keys:
        rolling_wmape_tfm       : np.ndarray
        rolling_wmape_chr       : np.ndarray
        rolling_mape_tfm        : np.ndarray
        rolling_mape_chr        : np.ndarray
        rolling_timestamps      : list  (aligned with rolling arrays)
        errors_tfm              : np.ndarray  (absolute errors)
        errors_chr              : np.ndarray
        spike_flags_tfm         : np.ndarray (bool)
        spike_flags_chr         : np.ndarray (bool)
        drift_info_tfm          : dict
        drift_info_chr          : dict
        stability_score_tfm     : float (0–100)
        stability_score_chr     : float (0–100)
        overall_wmape_tfm       : float
        overall_wmape_chr       : float
        overall_mape_tfm        : float
        overall_mape_chr        : float
        overall_mae_tfm         : float
        overall_mae_chr         : float
        overall_rmse_tfm        : float
        overall_rmse_chr        : float
        alerts_tfm              : list of alert dicts
        alerts_chr              : list of alert dicts
        n_steps                 : int
    """
    actual    = np.asarray(actual,    dtype=float)
    tfm_preds = np.asarray(tfm_preds, dtype=float)
    chr_preds = np.asarray(chr_preds, dtype=float)
    n         = len(actual)

    # Rolling metrics
    rw_tfm = rolling_wmape(actual, tfm_preds, window)
    rw_chr = rolling_wmape(actual, chr_preds, window)
    rm_tfm = rolling_mape(actual,  tfm_preds, window)
    rm_chr = rolling_mape(actual,  chr_preds, window)

    # Align timestamps with rolling arrays (last window-1 timestamps are valid)
    roll_len  = len(rw_tfm)
    roll_ts   = list(timestamps)[n - roll_len:]

    # Absolute errors
    err_tfm = np.abs(actual - tfm_preds)
    err_chr = np.abs(actual - chr_preds)

    # Spike detection
    spikes_tfm = detect_spikes(err_tfm, z_threshold=spike_z)
    spikes_chr = detect_spikes(err_chr, z_threshold=spike_z)

    # Drift
    drift_tfm = detect_drift(actual, tfm_preds, threshold=drift_threshold)
    drift_chr = detect_drift(actual, chr_preds, threshold=drift_threshold)

    # Stability
    stab_tfm = forecast_stability_score(rw_tfm)
    stab_chr = forecast_stability_score(rw_chr)

    # Overall metrics
    ow_tfm = wmape(actual, tfm_preds)
    ow_chr = wmape(actual, chr_preds)
    om_tfm = mape(actual,  tfm_preds)
    om_chr = mape(actual,  chr_preds)

    # Alerts
    alerts_tfm = generate_alerts(drift_tfm, spikes_tfm, rw_tfm, wmape_threshold)
    alerts_chr = generate_alerts(drift_chr, spikes_chr, rw_chr, wmape_threshold)

    return {
        "rolling_wmape_tfm":   rw_tfm,
        "rolling_wmape_chr":   rw_chr,
        "rolling_mape_tfm":    rm_tfm,
        "rolling_mape_chr":    rm_chr,
        "rolling_timestamps":  roll_ts,
        "errors_tfm":          err_tfm,
        "errors_chr":          err_chr,
        "spike_flags_tfm":     spikes_tfm,
        "spike_flags_chr":     spikes_chr,
        "drift_info_tfm":      drift_tfm,
        "drift_info_chr":      drift_chr,
        "stability_score_tfm": stab_tfm,
        "stability_score_chr": stab_chr,
        "overall_wmape_tfm":   round(ow_tfm, 2),
        "overall_wmape_chr":   round(ow_chr, 2),
        "overall_mape_tfm":    round(om_tfm, 2),
        "overall_mape_chr":    round(om_chr, 2),
        "overall_mae_tfm":     round(mae(actual, tfm_preds), 2),
        "overall_mae_chr":     round(mae(actual, chr_preds), 2),
        "overall_rmse_tfm":    round(rmse(actual, tfm_preds), 2),
        "overall_rmse_chr":    round(rmse(actual, chr_preds), 2),
        "alerts_tfm":          alerts_tfm,
        "alerts_chr":          alerts_chr,
        "n_steps":             n,
    }


def monitoring_summary_to_df(summary: dict, sid: str) -> pd.DataFrame:
    """Convert monitoring summary to a flat DataFrame for CSV export."""
    return pd.DataFrame([{
        "series_id":           sid,
        "n_steps":             summary["n_steps"],
        "overall_wmape_tfm":   summary["overall_wmape_tfm"],
        "overall_wmape_chr":   summary["overall_wmape_chr"],
        "overall_mape_tfm":    summary["overall_mape_tfm"],
        "overall_mape_chr":    summary["overall_mape_chr"],
        "overall_mae_tfm":     summary["overall_mae_tfm"],
        "overall_mae_chr":     summary["overall_mae_chr"],
        "overall_rmse_tfm":    summary["overall_rmse_tfm"],
        "overall_rmse_chr":    summary["overall_rmse_chr"],
        "stability_tfm":       summary["stability_score_tfm"],
        "stability_chr":       summary["stability_score_chr"],
        "drift_detected_tfm":  summary["drift_info_tfm"]["drift_detected"],
        "drift_detected_chr":  summary["drift_info_chr"]["drift_detected"],
        "n_spikes_tfm":        int(summary["spike_flags_tfm"].sum()),
        "n_spikes_chr":        int(summary["spike_flags_chr"].sum()),
    }])
