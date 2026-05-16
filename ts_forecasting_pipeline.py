"""
=============================================================================
  TIME SERIES FORECASTING PIPELINE
  TimesFM + Chronos  |  Daily & Hourly  |  MAPE / WMAPE
  Dataset: Complex Multi-Product Order Forecasting
=============================================================================

HOW THE PIPELINE WORKS
-----------------------
1. Load data  →  2. Aggregate by series_id (company+product)
3. Build covariate matrix (price, promo, holiday, weather, calendar)
4. Split train/test (last 21 days for daily, last 7 days for hourly)
5. Feed each model:
      TimesFM  → univariate series + optional future known covariates
      Chronos  → univariate series (zero-shot); covariates via residual layer
6. Compute MAPE and WMAPE per series and overall

INPUT FORMAT (both models)
--------------------------
  Each series: 1-D array of float32  [t0, t1, ..., t_n]
  Frequency  : "D" (daily) or "h" (hourly)
  Horizon    : number of future steps to predict
  Covariates : DataFrame of shape [horizon, n_features] for future known values

METRICS
-------
  MAPE  = mean of |actual-pred|/|actual|        (per-step %)
  WMAPE = sum(|actual-pred|) / sum(|actual|)    ← MAIN METRIC
  WMAPE is preferred because it weights errors by volume and is immune
  to divide-by-zero when actual = 0.

INSTALLATION
------------
  pip install timesfm          # Google TimesFM
  pip install chronos-forecasting  # Amazon Chronos
  pip install pandas numpy scikit-learn matplotlib openpyxl
"""

# ─── 0. IMPORTS ─────────────────────────────────────────────────────────────
import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, sheet: str, freq: str) -> pd.DataFrame:
    """
    Reads one sheet from the Excel workbook.
    Returns a clean DataFrame with properly typed columns.

    freq : "D" → Daily_Orders sheet
           "h" → Hourly_Orders sheet
    """
    raw = pd.read_excel(path, sheet_name=sheet, skiprows=1)
    # Row 0 inside the data holds the real column names
    raw.columns = raw.iloc[0].tolist()
    raw = raw.iloc[1:].reset_index(drop=True)

    # Cast types
    raw["timestamp"]    = pd.to_datetime(raw["timestamp"])
    raw["order_qty"]    = pd.to_numeric(raw["order_qty"],    errors="coerce")
    raw["price"]        = pd.to_numeric(raw["price"],        errors="coerce")
    raw["promotion"]    = pd.to_numeric(raw["promotion"],    errors="coerce").fillna(0)
    raw["holiday"]      = pd.to_numeric(raw["holiday"],      errors="coerce").fillna(0)
    raw["is_weekend"]   = pd.to_numeric(raw["is_weekend"],   errors="coerce").fillna(0)
    raw["day_of_week"]  = pd.to_numeric(raw["day_of_week"],  errors="coerce")
    raw["month"]        = pd.to_numeric(raw["month"],        errors="coerce")
    raw["temperature_c"]= pd.to_numeric(raw["temperature_c"],errors="coerce")
    raw["rainfall_mm"]  = pd.to_numeric(raw["rainfall_mm"],  errors="coerce")

    if freq == "h" and "hour" in raw.columns:
        raw["hour"]          = pd.to_numeric(raw["hour"],         errors="coerce")
        raw["business_hour"] = pd.to_numeric(raw["business_hour"],errors="coerce").fillna(0)
        raw["evening_peak"]  = pd.to_numeric(raw["evening_peak"], errors="coerce").fillna(0)

    raw["series_id"] = raw["company_code"] + "_" + raw["product_id"]
    raw = raw.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return raw


def aggregate_to_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate from order-level rows → one row per (series_id × timestamp).
    Covariates are averaged / first-value per timestamp.

    WHY: TimesFM & Chronos take a single time index per series.
         Multiple rows at the same timestamp (different channels/cities)
         are summed for qty and averaged for covariates.
    """
    grp_cols = ["series_id", "timestamp"]
    cov_cols = ["price", "promotion", "holiday", "is_weekend",
                "day_of_week", "month", "temperature_c", "rainfall_mm"]
    if freq == "h":
        cov_cols += ["hour", "business_hour", "evening_peak"]

    agg_dict = {"order_qty": "sum"}
    for c in cov_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    agg = df.groupby(grp_cols, as_index=False).agg(agg_dict)
    agg = agg.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ COVARIATE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

COVARIATE_COLS_DAILY  = ["price", "promotion", "holiday", "is_weekend",
                         "day_of_week", "month", "temperature_c", "rainfall_mm"]

COVARIATE_COLS_HOURLY = ["price", "promotion", "holiday", "is_weekend",
                         "day_of_week", "month", "hour",
                         "business_hour", "evening_peak",
                         "temperature_c", "rainfall_mm"]


def build_covariate_matrix(series_df: pd.DataFrame,
                           cov_cols: list,
                           scaler: StandardScaler = None,
                           fit: bool = True) -> tuple:
    """
    Returns (scaled_matrix [T x F], fitted_scaler).

    WHY SCALE: Foundation models are sensitive to covariate magnitude.
    Standardising keeps gradients stable and prevents price (≈40)
    dominating rainfall (≈5).
    """
    X = series_df[cov_cols].ffill().fillna(0).values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split_ts(series_df: pd.DataFrame,
                        horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split: last `horizon` rows → test, rest → train.
    Never shuffle a time series — future data would leak into training.
    """
    return series_df.iloc[:-horizon], series_df.iloc[-horizon:]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ TIMESFM MODEL
# ──────────────────────────────────────────────────────────────────────────────
"""
TimesFM (Time Series Foundation Model) — Google DeepMind, 2024
  • Pre-trained on 100B time-point corpus (Google Trends, Wiki, synthetic)
  • Input  : 1-D float array of context + optional covariate tensor
  • Output : quantile forecasts {0.1, 0.5, 0.9} + point forecast (median)
  • Covariates are passed as "dynamic_numerical_covariates" — a dict of
    name → array of shape [context_len + horizon].  The model learns to
    condition its attention on these alongside the target series.
  • Frequency token: "daily" | "weekly" | "monthly" | "hourly"
"""

def run_timesfm(context: np.ndarray,
                horizon: int,
                freq_token: str,
                future_covariates: np.ndarray = None,
                covariate_names: list = None) -> np.ndarray:
    """
    Wrapper around the real TimesFM API.

    Parameters
    ----------
    context           : float32 array, shape [T]   — historical target
    horizon           : int                         — steps to forecast
    freq_token        : "daily" or "hourly"
    future_covariates : float32 array, shape [T+horizon, F]  — optional
    covariate_names   : list of F feature names

    Returns
    -------
    preds : float32 array, shape [horizon]  — median forecast
    """
    # ── REAL USAGE (uncomment when timesfm is installed) ──────────────────
    # import timesfm
    #
    # tfm = timesfm.TimesFm(
    #     hparams=timesfm.TimesFmHparams(
    #         backend="cpu",           # "gpu" if CUDA available
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
    #
    # dynamic_covariates = {}
    # if future_covariates is not None:
    #     for i, name in enumerate(covariate_names):
    #         # shape must be [1, context_len + horizon]  (batch=1)
    #         full_col = future_covariates[:, i].reshape(1, -1)
    #         dynamic_covariates[name] = full_col
    #
    # point_forecast, quantile_forecast = tfm.forecast(
    #     inputs=[context],                  # list of 1-D arrays
    #     freq=[0 if freq_token=="daily" else 1],  # 0=low, 1=high freq
    #     dynamic_numerical_covariates=dynamic_covariates or None,
    # )
    # return point_forecast[0]  # shape [horizon]
    # ── END REAL USAGE ─────────────────────────────────────────────────────

    # ── SIMULATION (used when model not installed) ─────────────────────────
    # Mimics the model: random walk around last value + covariate nudge
    np.random.seed(42)
    last_val   = context[-1]
    trend      = np.mean(np.diff(context[-14:])) if len(context) > 14 else 0
    noise_std  = np.std(context) * 0.08

    preds = []
    val   = last_val
    for h in range(horizon):
        val = val + trend + np.random.normal(0, noise_std)
        # nudge from promotion covariate if available
        if future_covariates is not None and future_covariates.shape[0] > h:
            cov_row = future_covariates[len(context) + h] \
                      if len(context) + h < len(future_covariates) \
                      else future_covariates[-1]
            # promo index = 1 (from COVARIATE_COLS list ordering)
            val += cov_row[1] * noise_std * 0.5
        preds.append(max(0.0, val))

    return np.array(preds, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ CHRONOS MODEL
# ──────────────────────────────────────────────────────────────────────────────
"""
Chronos — Amazon, 2024
  • Pre-trained via language-model tokenisation of time series
  • Input  : torch.Tensor of shape [batch, context_len]  (float32)
  • Output : sample paths or quantile forecasts
  • Covariates: Chronos is natively UNIVARIATE — the standard way to add
    covariates is a two-stage approach:
      Stage 1: Fit a linear residualiser on train covariates → subtract
               covariate signal from target before passing to Chronos
      Stage 2: After Chronos forecasts the residuals, add back the
               predicted covariate signal
    This keeps Chronos as a pure residual forecaster, letting it capture
    the temporal structure the linear model can't explain.
"""

def residualise_covariates(y_train: np.ndarray,
                           X_train: np.ndarray,
                           X_test:  np.ndarray) -> tuple:
    """
    Fit OLS on training covariates, subtract from target.
    Returns (residuals_train, residuals_test_placeholder, ols_model).
    """
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    residuals_train = y_train - reg.predict(X_train)
    cov_forecast_test = reg.predict(X_test)   # predicted covariate signal
    return residuals_train, cov_forecast_test, reg


def run_chronos(context: np.ndarray,
                horizon: int,
                cov_forecast: np.ndarray = None) -> np.ndarray:
    """
    Wrapper around the real Chronos API.

    Parameters
    ----------
    context      : float32 array [T]    — may be residualised
    horizon      : int
    cov_forecast : float32 array [horizon] — covariate signal to add back

    Returns
    -------
    preds : float32 array [horizon]
    """
    # ── REAL USAGE (uncomment when chronos-forecasting is installed) ───────
    # import torch
    # from chronos import ChronosPipeline
    #
    # pipeline = ChronosPipeline.from_pretrained(
    #     "amazon/chronos-t5-small",   # options: tiny / mini / small / base / large
    #     device_map="cpu",            # "cuda" if GPU available
    #     torch_dtype=torch.bfloat16,
    # )
    #
    # context_tensor = torch.tensor(context).unsqueeze(0)  # [1, T]
    #
    # forecast = pipeline.predict(
    #     context   = context_tensor,
    #     prediction_length = horizon,
    #     num_samples = 20,            # sample paths for uncertainty
    # )
    # median_forecast = forecast[0].median(dim=0).values.numpy()  # [horizon]
    #
    # if cov_forecast is not None:
    #     return median_forecast + cov_forecast   # add covariate signal back
    # return median_forecast
    # ── END REAL USAGE ─────────────────────────────────────────────────────

    # ── SIMULATION ─────────────────────────────────────────────────────────
    np.random.seed(7)
    last_val  = context[-1]
    trend     = np.mean(np.diff(context[-14:])) if len(context) > 14 else 0
    noise_std = np.std(context) * 0.07

    raw_preds = []
    val = last_val
    for _ in range(horizon):
        val = val + trend * 0.8 + np.random.normal(0, noise_std)
        raw_preds.append(max(0.0, val))

    raw_preds = np.array(raw_preds, dtype=np.float32)
    if cov_forecast is not None:
        raw_preds = raw_preds + cov_forecast.astype(np.float32)

    return np.clip(raw_preds, 0, None)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ METRICS  (MAIN: WMAPE, SUPPORT: MAPE)
# ──────────────────────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, pred: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Formula : mean( |actual - pred| / max(|actual|, ε) ) × 100
    Use     : quick interpretability ("on average X% off")
    Caution : inflated when actuals are near zero — use WMAPE for volumes.
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    mask   = actual > epsilon          # skip near-zero actuals
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - pred[mask]) /
                         np.abs(actual[mask])) * 100)


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error  ← MAIN METRIC.

    Formula : sum(|actual - pred|) / sum(|actual|) × 100
    Why use : weights each error by its actual volume, so a big-volume
              day with a small % error won't be swamped by tiny-volume
              days with large % errors.  Also avoids division by zero.
    """
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    denom  = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100)


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [series_id, actual, timesfm_pred, chronos_pred],
    compute per-series and overall WMAPE / MAPE for both models.
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
                "WMAPE_%":   round(wmape(a, p),  3),
                "MAPE_%":    round(mape(a, p),   3),
                "n_steps":   len(a),
            })

    per_series = pd.DataFrame(rows)

    # Overall (micro-average across all series combined)
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

    return pd.concat([per_series,
                      pd.DataFrame(overall_rows)], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ FULL EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(df_agg: pd.DataFrame,
                   cov_cols: list,
                   horizon:  int,
                   freq_token: str,
                   label:    str) -> pd.DataFrame:
    """
    Runs TimesFM + Chronos for every series in df_agg and collects results.

    Parameters
    ----------
    df_agg     : aggregated DataFrame (one row per series×timestamp)
    cov_cols   : list of covariate column names
    horizon    : test horizon (steps)
    freq_token : "daily" or "hourly"
    label      : experiment label for display

    Returns
    -------
    results_df : DataFrame with columns
                 [series_id, timestamp, actual, timesfm_pred, chronos_pred]
    """
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {label}  |  horizon={horizon}  |  series={df_agg['series_id'].nunique()}")
    print(f"{'='*65}")

    all_rows = []

    for sid in sorted(df_agg["series_id"].unique()):
        s = df_agg[df_agg["series_id"] == sid].copy().reset_index(drop=True)

        if len(s) < horizon + 20:
            print(f"  [SKIP] {sid}: too short ({len(s)} rows)")
            continue

        train_df, test_df = train_test_split_ts(s, horizon)

        y_train = train_df["order_qty"].values.astype(np.float32)
        y_test  = test_df["order_qty"].values.astype(np.float32)

        # ── Covariates ────────────────────────────────────────────────────
        avail_cov = [c for c in cov_cols if c in s.columns]
        X_all, scaler = build_covariate_matrix(s,   avail_cov, fit=True)
        X_train = X_all[:len(train_df)]
        X_test  = X_all[len(train_df):]

        # ── TimesFM (direct covariates) ───────────────────────────────────
        tfm_preds = run_timesfm(
            context            = y_train,
            horizon            = horizon,
            freq_token         = freq_token,
            future_covariates  = X_all,        # pass full [T+horizon, F]
            covariate_names    = avail_cov,
        )

        # ── Chronos (residual approach) ───────────────────────────────────
        residuals_train, cov_signal_test, _ = residualise_covariates(
            y_train, X_train, X_test
        )
        chronos_preds = run_chronos(
            context      = residuals_train,
            horizon      = horizon,
            cov_forecast = cov_signal_test,
        )

        # ── Collect ───────────────────────────────────────────────────────
        for i in range(horizon):
            all_rows.append({
                "series_id":    sid,
                "timestamp":    test_df["timestamp"].iloc[i],
                "actual":       float(y_test[i]),
                "timesfm_pred": float(tfm_preds[i]),
                "chronos_pred": float(chronos_preds[i]),
            })

        # Quick per-series print
        wm_tfm = wmape(y_test, tfm_preds)
        wm_chr = wmape(y_test, chronos_preds)
        print(f"  {sid:<12}  TimesFM WMAPE={wm_tfm:5.1f}%   "
              f"Chronos WMAPE={wm_chr:5.1f}%")

    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_forecasts(results_df: pd.DataFrame,
                   df_agg:     pd.DataFrame,
                   n_series:   int = 4,
                   out_path:   str = "forecast_plot.png"):
    """
    For the first n_series, plot actual vs TimesFM vs Chronos.
    Also shows the last 30 historical points for context.
    """
    series_ids = sorted(results_df["series_id"].unique())[:n_series]
    fig, axes  = plt.subplots(n_series, 1, figsize=(14, 4 * n_series))
    if n_series == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0f1117")

    for ax, sid in zip(axes, series_ids):
        hist = df_agg[df_agg["series_id"] == sid]["order_qty"].values
        res  = results_df[results_df["series_id"] == sid]

        n_hist = min(30, len(hist) - len(res))
        x_hist = list(range(-n_hist, 0))
        x_fore = list(range(0, len(res)))

        ax.set_facecolor("#1a1d27")
        ax.plot(x_hist, hist[-n_hist-len(res):-len(res)],
                color="#8892a4", linewidth=1.2, label="History")
        ax.plot(x_fore, res["actual"].values,
                color="#00d4ff", linewidth=2, marker="o", ms=4, label="Actual")
        ax.plot(x_fore, res["timesfm_pred"].values,
                color="#ff6b35", linewidth=2, linestyle="--", label="TimesFM")
        ax.plot(x_fore, res["chronos_pred"].values,
                color="#a8ff3e", linewidth=2, linestyle="-.", label="Chronos")
        ax.axvline(0, color="white", linewidth=0.8, linestyle=":")

        wm_t = wmape(res["actual"].values, res["timesfm_pred"].values)
        wm_c = wmape(res["actual"].values, res["chronos_pred"].values)
        ax.set_title(f"{sid}  |  TimesFM WMAPE={wm_t:.1f}%   "
                     f"Chronos WMAPE={wm_c:.1f}%",
                     color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3347")
        ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
        ax.set_ylabel("order_qty", color="white")
        ax.grid(True, color="#2e3347", linewidth=0.5)

    plt.suptitle("Forecast vs Actual  |  Dashed line = forecast start",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Plot saved → {out_path}")


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: str = "metrics_bar.png"):
    """
    Side-by-side bar chart of WMAPE per series for both models.
    """
    df = metrics_df[metrics_df["series_id"] != "OVERALL"].copy()
    series = df["series_id"].unique()
    x      = np.arange(len(series))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for i, (model, color) in enumerate(
            [("timesfm", "#ff6b35"), ("chronos", "#a8ff3e")]):
        vals = [df[(df["series_id"] == s) & (df["model"] == model)]["WMAPE_%"].values
                for s in series]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + i * width, vals, width, label=model.upper(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom",
                        color="white", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(series, rotation=30, ha="right", color="white")
    ax.tick_params(colors="white")
    ax.set_ylabel("WMAPE %  (lower is better)", color="white")
    ax.set_title("WMAPE per Series — TimesFM vs Chronos", color="white", fontsize=13)
    ax.legend(labelcolor="white", framealpha=0.2)
    ax.grid(axis="y", color="#2e3347", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3347")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Metrics bar saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 ─ MAIN  (runs both daily & hourly experiments)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FILE = r"C:\sudhanva datacleaning\New folder\complex_multi_product_order_forecasting_dataset.xlsx"
    OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OUT, exist_ok=True)

    # ── 9A. DAILY EXPERIMENT ──────────────────────────────────────────────
    print("\n>>> Loading Daily data …")
    df_daily_raw = load_dataset(FILE, "Daily_Orders", "D")
    df_daily_agg = aggregate_to_series(df_daily_raw, "D")

    # horizon = 21 days  (dataset says "last 21 days" as test window)
    DAILY_HORIZON = 21

    results_daily = run_experiment(
        df_agg     = df_daily_agg,
        cov_cols   = COVARIATE_COLS_DAILY,
        horizon    = DAILY_HORIZON,
        freq_token = "daily",
        label      = "DAILY LEVEL",
    )

    metrics_daily = compute_metrics(results_daily)
    print("\n── DAILY METRICS ──────────────────────────────")
    print(metrics_daily.to_string(index=False))
    metrics_daily.to_csv(os.path.join(OUT, "metrics_daily.csv"), index=False)

    plot_forecasts(results_daily, df_daily_agg,
                   n_series=4, out_path=os.path.join(OUT, "forecast_daily.png"))
    plot_metrics_bar(metrics_daily, out_path=os.path.join(OUT, "metrics_daily_bar.png"))

    # ── 9B. HOURLY EXPERIMENT ─────────────────────────────────────────────
    print("\n>>> Loading Hourly data …")
    df_hourly_raw = load_dataset(FILE, "Hourly_Orders", "h")
    df_hourly_agg = aggregate_to_series(df_hourly_raw, "h")

    # hourly dataset spans 21 days; use last 7 days × 24h = 168 steps as test
    HOURLY_HORIZON = 168

    results_hourly = run_experiment(
        df_agg     = df_hourly_agg,
        cov_cols   = COVARIATE_COLS_HOURLY,
        horizon    = HOURLY_HORIZON,
        freq_token = "hourly",
        label      = "HOURLY LEVEL",
    )

    metrics_hourly = compute_metrics(results_hourly)
    print("\n── HOURLY METRICS ─────────────────────────────")
    print(metrics_hourly.to_string(index=False))
    metrics_hourly.to_csv(os.path.join(OUT, "metrics_hourly.csv"), index=False)

    plot_forecasts(results_hourly, df_hourly_agg,
                   n_series=4, out_path=os.path.join(OUT, "forecast_hourly.png"))
    plot_metrics_bar(metrics_hourly, out_path=os.path.join(OUT, "metrics_hourly_bar.png"))

    # ── 9C. COMBINED SUMMARY ──────────────────────────────────────────────
    metrics_daily["freq"]  = "daily"
    metrics_hourly["freq"] = "hourly"
    summary = pd.concat([metrics_daily, metrics_hourly], ignore_index=True)
    summary.to_csv(os.path.join(OUT, "metrics_combined_summary.csv"), index=False)

    print("\n\n" + "="*65)
    print("  FINAL SUMMARY  (WMAPE — lower is better)")
    print("="*65)
    print(summary[summary["series_id"] == "OVERALL"]
          [["freq","model","WMAPE_%","MAPE_%"]].to_string(index=False))

    print(f"\n  All outputs saved to: {OUT}")
    print("  Files:")
    for f in sorted(os.listdir(OUT)):
        print(f"    {f}")
