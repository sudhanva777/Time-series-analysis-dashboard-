"""
forecasting/models.py
─────────────────────
TimesFM and Chronos model wrappers.

Currently uses simulation mode (no GPU/heavy dependencies needed).
To use the real models, uncomment the marked blocks and install:
  pip install timesfm
  pip install chronos-forecasting torch

Each wrapper returns:
  preds           : float32 array [horizon]     — point forecast
  lower_bound     : float32 array [horizon]     — ~10th percentile estimate
  upper_bound     : float32 array [horizon]     — ~90th percentile estimate
"""

import numpy as np
from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────────────────────
# TimesFM
# ─────────────────────────────────────────────────────────────────────────────

def run_timesfm(
    context: np.ndarray,
    horizon: int,
    freq_token: str,
    future_covariates: np.ndarray = None,
    covariate_names: list = None,
) -> tuple:
    """
    TimesFM wrapper — Google DeepMind, 2024.

    Parameters
    ----------
    context           : float32 [T]            historical target values
    horizon           : int                    steps to forecast
    freq_token        : "daily" or "hourly"
    future_covariates : float32 [T+horizon, F] optional covariate matrix
    covariate_names   : list of F feature names

    Returns
    -------
    (preds, lower_bound, upper_bound) — each float32 [horizon]
    """

    # ── REAL USAGE (uncomment when timesfm is installed) ─────────────────────
    # import timesfm
    # tfm = timesfm.TimesFm(
    #     hparams=timesfm.TimesFmHparams(
    #         backend="cpu",
    #         per_core_batch_size=32,
    #         horizon_len=horizon,
    #         num_layers=20,
    #         model_dims=1280,
    #         use_positional_embedding=False,
    #     ),
    #     checkpoint=timesfm.TimesFmCheckpoint(
    #         huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    #     ),
    # )
    # dynamic_covariates = {}
    # if future_covariates is not None and covariate_names:
    #     for i, name in enumerate(covariate_names):
    #         dynamic_covariates[name] = future_covariates[:, i].reshape(1, -1)
    # point_forecast, quantile_forecast = tfm.forecast(
    #     inputs=[context],
    #     freq=[0 if freq_token == "daily" else 1],
    #     dynamic_numerical_covariates=dynamic_covariates or None,
    # )
    # preds = point_forecast[0].astype(np.float32)
    # # quantile_forecast shape: [1, horizon, n_quantiles]  quantiles=[0.1, 0.5, 0.9]
    # lower_bound = quantile_forecast[0, :, 0].astype(np.float32)
    # upper_bound = quantile_forecast[0, :, 2].astype(np.float32)
    # return preds, lower_bound, upper_bound
    # ── END REAL USAGE ────────────────────────────────────────────────────────

    # ── SIMULATION ────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    last_val  = float(context[-1])
    trend     = float(np.mean(np.diff(context[-14:]))) if len(context) > 14 else 0.0
    noise_std = float(np.std(context)) * 0.08 if np.std(context) > 0 else 1.0

    preds = []
    val   = last_val
    for h in range(horizon):
        val = val + trend + rng.normal(0, noise_std)
        if future_covariates is not None:
            idx = min(len(context) + h, len(future_covariates) - 1)
            val += future_covariates[idx][1] * noise_std * 0.5  # promo nudge
        preds.append(max(0.0, val))

    preds       = np.array(preds, dtype=np.float32)
    ci_width    = noise_std * 1.65  # ~90% CI
    lower_bound = np.clip(preds - ci_width, 0, None)
    upper_bound = preds + ci_width

    return preds, lower_bound.astype(np.float32), upper_bound.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Chronos
# ─────────────────────────────────────────────────────────────────────────────

def residualise_covariates(
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    Fit Ridge regression on training covariates, subtract from target.
    Stage 1 of the Chronos two-stage covariate approach.

    Returns
    -------
    (residuals_train, cov_forecast_test, ridge_model)
    """
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    residuals_train   = y_train - reg.predict(X_train)
    cov_forecast_test = reg.predict(X_test)
    return residuals_train, cov_forecast_test, reg


def run_chronos(
    context: np.ndarray,
    horizon: int,
    cov_forecast: np.ndarray = None,
) -> tuple:
    """
    Chronos wrapper — Amazon, 2024.

    Chronos is natively univariate; covariates are added via the
    residualise_covariates() two-stage approach (subtract before, add after).

    Parameters
    ----------
    context      : float32 [T]       may be residualised
    horizon      : int
    cov_forecast : float32 [horizon] covariate signal to add back

    Returns
    -------
    (preds, lower_bound, upper_bound) — each float32 [horizon]
    """

    # ── REAL USAGE (uncomment when chronos-forecasting is installed) ──────────
    # import torch
    # from chronos import ChronosPipeline
    # pipeline = ChronosPipeline.from_pretrained(
    #     "amazon/chronos-t5-small",
    #     device_map="cpu",
    #     torch_dtype=torch.bfloat16,
    # )
    # context_tensor = torch.tensor(context).unsqueeze(0)
    # forecast = pipeline.predict(
    #     context=context_tensor,
    #     prediction_length=horizon,
    #     num_samples=20,
    # )
    # # forecast shape: [1, num_samples, horizon]
    # samples   = forecast[0].numpy()  # [num_samples, horizon]
    # preds       = np.median(samples, axis=0).astype(np.float32)
    # lower_bound = np.percentile(samples, 10, axis=0).astype(np.float32)
    # upper_bound = np.percentile(samples, 90, axis=0).astype(np.float32)
    # if cov_forecast is not None:
    #     preds       = preds       + cov_forecast
    #     lower_bound = lower_bound + cov_forecast
    #     upper_bound = upper_bound + cov_forecast
    # return np.clip(preds, 0, None), np.clip(lower_bound, 0, None), upper_bound
    # ── END REAL USAGE ────────────────────────────────────────────────────────

    # ── SIMULATION ────────────────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    last_val  = float(context[-1])
    trend     = float(np.mean(np.diff(context[-14:]))) if len(context) > 14 else 0.0
    noise_std = float(np.std(context)) * 0.07 if np.std(context) > 0 else 1.0

    raw_preds = []
    val = last_val
    for _ in range(horizon):
        val = val + trend * 0.8 + rng.normal(0, noise_std)
        raw_preds.append(max(0.0, val))

    raw_preds = np.array(raw_preds, dtype=np.float32)

    if cov_forecast is not None:
        raw_preds = raw_preds + cov_forecast.astype(np.float32)

    preds       = np.clip(raw_preds, 0, None)
    ci_width    = noise_std * 1.65
    lower_bound = np.clip(preds - ci_width, 0, None).astype(np.float32)
    upper_bound = (preds + ci_width).astype(np.float32)

    return preds, lower_bound, upper_bound
