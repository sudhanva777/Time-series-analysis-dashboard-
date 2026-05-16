"""
forecasting/pipeline.py
───────────────────────
Experiment runner — orchestrates preprocessing → covariates → models → metrics.

Provides:
  run_single_series()  — one series, returns dict with preds + confidence bands
  run_experiment()     — all series in a DataFrame, returns results DataFrame
"""

import numpy as np
import pandas as pd

from forecasting.preprocessing import train_test_split_ts
from forecasting.covariates    import build_covariate_matrix
from forecasting.models        import run_timesfm, run_chronos, residualise_covariates
from forecasting.metrics       import wmape


MIN_SERIES_LENGTH_BUFFER = 20  # series must have at least horizon + this many rows


def run_single_series(
    series_df: pd.DataFrame,
    horizon:   int,
    freq_token: str,
    cov_cols:  list,
    sid:       str,
) -> dict:
    """
    Run TimesFM + Chronos on a single series and return a result dict.

    Parameters
    ----------
    series_df  : DataFrame with one series (columns: timestamp, order_qty, covariates)
    horizon    : number of steps to forecast
    freq_token : "daily" or "hourly"
    cov_cols   : list of covariate column names
    sid        : series identifier (for labelling)

    Returns
    -------
    dict with keys:
        series_id            : str
        timestamp            : list of timestamps (test period)
        actual               : list[float]
        timesfm_pred         : list[float]
        timesfm_lower        : list[float]
        timesfm_upper        : list[float]
        chronos_pred         : list[float]
        chronos_lower        : list[float]
        chronos_upper        : list[float]
        train_length         : int
        wmape_timesfm        : float
        wmape_chronos        : float
    """
    series_df = series_df.sort_values("timestamp").reset_index(drop=True)

    if len(series_df) < horizon + MIN_SERIES_LENGTH_BUFFER:
        raise ValueError(
            f"Series '{sid}' is too short ({len(series_df)} rows) "
            f"for horizon={horizon}. Need at least {horizon + MIN_SERIES_LENGTH_BUFFER}."
        )

    train_df, test_df = train_test_split_ts(series_df, horizon)

    y_train = train_df["order_qty"].values.astype(np.float32)
    y_test  = test_df["order_qty"].values.astype(np.float32)

    # Covariates
    avail_cov = [c for c in cov_cols if c in series_df.columns]
    X_all, _  = build_covariate_matrix(series_df, avail_cov, fit=True)
    X_train   = X_all[:len(train_df)]
    X_test    = X_all[len(train_df):]

    # TimesFM
    tfm_preds, tfm_lower, tfm_upper = run_timesfm(
        context           = y_train,
        horizon           = horizon,
        freq_token        = freq_token,
        future_covariates = X_all,
        covariate_names   = avail_cov,
    )

    # Chronos (residual approach)
    residuals_train, cov_signal_test, _ = residualise_covariates(
        y_train, X_train, X_test
    )
    chr_preds, chr_lower, chr_upper = run_chronos(
        context      = residuals_train,
        horizon      = horizon,
        cov_forecast = cov_signal_test,
    )

    return {
        "series_id":     sid,
        "timestamp":     test_df["timestamp"].tolist(),
        "actual":        y_test.tolist(),
        "timesfm_pred":  tfm_preds.tolist(),
        "timesfm_lower": tfm_lower.tolist(),
        "timesfm_upper": tfm_upper.tolist(),
        "chronos_pred":  chr_preds.tolist(),
        "chronos_lower": chr_lower.tolist(),
        "chronos_upper": chr_upper.tolist(),
        "train_length":  len(train_df),
        "wmape_timesfm": round(wmape(y_test, tfm_preds), 2),
        "wmape_chronos": round(wmape(y_test, chr_preds), 2),
    }


def run_experiment(
    df_agg:     pd.DataFrame,
    cov_cols:   list,
    horizon:    int,
    freq_token: str,
    label:      str = "EXPERIMENT",
    progress_cb = None,
) -> pd.DataFrame:
    """
    Run TimesFM + Chronos for every series in df_agg.

    Parameters
    ----------
    df_agg       : aggregated DataFrame (one row per series×timestamp)
    cov_cols     : list of covariate column names
    horizon      : test horizon (steps)
    freq_token   : "daily" or "hourly"
    label        : display label
    progress_cb  : optional callable(current, total, sid) for progress reporting

    Returns
    -------
    results_df : DataFrame with columns
                 [series_id, timestamp, actual,
                  timesfm_pred, timesfm_lower, timesfm_upper,
                  chronos_pred, chronos_lower, chronos_upper]
    """
    series_list = sorted(df_agg["series_id"].unique())
    total       = len(series_list)
    all_rows    = []
    skipped     = []

    for idx, sid in enumerate(series_list):
        if progress_cb:
            progress_cb(idx, total, sid)

        s = df_agg[df_agg["series_id"] == sid].copy().reset_index(drop=True)

        if len(s) < horizon + MIN_SERIES_LENGTH_BUFFER:
            skipped.append(sid)
            continue

        try:
            res = run_single_series(s, horizon, freq_token, cov_cols, sid)
        except Exception as e:
            skipped.append(f"{sid} (error: {e})")
            continue

        for i in range(horizon):
            all_rows.append({
                "series_id":     res["series_id"],
                "timestamp":     res["timestamp"][i],
                "actual":        res["actual"][i],
                "timesfm_pred":  res["timesfm_pred"][i],
                "timesfm_lower": res["timesfm_lower"][i],
                "timesfm_upper": res["timesfm_upper"][i],
                "chronos_pred":  res["chronos_pred"][i],
                "chronos_lower": res["chronos_lower"][i],
                "chronos_upper": res["chronos_upper"][i],
            })

    if progress_cb:
        progress_cb(total, total, "done")

    return pd.DataFrame(all_rows)
