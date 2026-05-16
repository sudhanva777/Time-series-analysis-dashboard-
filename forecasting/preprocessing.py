"""
forecasting/preprocessing.py
─────────────────────────────
Data loading, validation, and aggregation logic.
Supports XLSX (double-header format) and CSV (standard first-row headers).
"""

import warnings
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Required columns for the forecasting pipeline ────────────────────────────
REQUIRED_COLS = [
    "timestamp", "order_qty", "company_code", "product_id",
    "price", "promotion", "holiday", "is_weekend",
    "day_of_week", "month", "temperature_c", "rainfall_mm",
]

OPTIONAL_COLS_HOURLY = ["hour", "business_hour", "evening_peak"]

COVARIATE_COLS_DAILY = [
    "price", "promotion", "holiday", "is_weekend",
    "day_of_week", "month", "temperature_c", "rainfall_mm",
]

COVARIATE_COLS_HOURLY = [
    "price", "promotion", "holiday", "is_weekend",
    "day_of_week", "month", "hour",
    "business_hour", "evening_peak",
    "temperature_c", "rainfall_mm",
]


# ── File loading ──────────────────────────────────────────────────────────────

def smart_load(file_obj, freq: str, sheet: str = None) -> pd.DataFrame:
    """
    Unified loader for XLSX and CSV files.

    XLSX: Uses double-header format (row 0 = blank section title, row 1 = real columns).
          Sheet defaults to 'Daily_Orders' or 'Hourly_Orders' based on freq.
    CSV:  Standard first-row headers.

    Parameters
    ----------
    file_obj : file path (str) or file-like object (Streamlit UploadedFile)
    freq     : "D" (daily) or "h" (hourly)
    sheet    : optional explicit sheet name (XLSX only)

    Returns
    -------
    pd.DataFrame with typed columns and series_id added
    """
    # Determine file type
    if hasattr(file_obj, "name"):
        fname = file_obj.name.lower()
    else:
        fname = str(file_obj).lower()

    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        if sheet is None:
            sheet = "Daily_Orders" if freq == "D" else "Hourly_Orders"
        return _load_xlsx(file_obj, sheet, freq)
    elif fname.endswith(".csv"):
        return _load_csv(file_obj, freq)
    else:
        raise ValueError(f"Unsupported file format. Please upload .xlsx or .csv files.")


def _load_xlsx(file_obj, sheet: str, freq: str) -> pd.DataFrame:
    """Load Excel file with double-header format."""
    try:
        raw = pd.read_excel(file_obj, sheet_name=sheet, skiprows=1)
    except Exception as e:
        raise ValueError(f"Could not read sheet '{sheet}' from Excel file: {e}")

    if raw.empty:
        raise ValueError(f"Sheet '{sheet}' is empty.")

    # Row 0 inside the data holds the real column names
    raw.columns = raw.iloc[0].tolist()
    raw = raw.iloc[1:].reset_index(drop=True)

    if raw.empty:
        raise ValueError(f"Sheet '{sheet}' has no data rows after header extraction.")

    return _cast_types(raw, freq)


def _load_csv(file_obj, freq: str) -> pd.DataFrame:
    """Load CSV file with standard first-row headers."""
    try:
        raw = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    if raw.empty:
        raise ValueError("CSV file is empty.")

    return _cast_types(raw, freq)


def _cast_types(raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Type-cast columns and build series_id."""
    # Normalize column names
    raw.columns = [str(c).strip() for c in raw.columns]

    numeric_cols = [
        "order_qty", "price", "promotion", "holiday",
        "is_weekend", "day_of_week", "month",
        "temperature_c", "rainfall_mm",
    ]
    fill_zero_cols = ["promotion", "holiday", "is_weekend"]

    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")

    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            if col in fill_zero_cols:
                raw[col] = raw[col].fillna(0)

    if freq == "h":
        for col in OPTIONAL_COLS_HOURLY:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
                if col in ["business_hour", "evening_peak"]:
                    raw[col] = raw[col].fillna(0)

    # Build series_id from company_code + product_id
    if "company_code" in raw.columns and "product_id" in raw.columns:
        raw["series_id"] = (
            raw["company_code"].astype(str) + "_" + raw["product_id"].astype(str)
        )

    raw = raw.sort_values(
        ["series_id", "timestamp"] if "series_id" in raw.columns else ["timestamp"]
    ).reset_index(drop=True)

    return raw


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, freq: str = "D") -> dict:
    """
    Validate a loaded DataFrame for use in the forecasting pipeline.

    Returns
    -------
    dict with keys:
        valid         : bool
        errors        : list of blocking error strings
        warnings      : list of non-blocking warning strings
        info          : dict of dataset statistics
    """
    errors = []
    warnings_list = []

    if df is None or df.empty:
        return {"valid": False, "errors": ["DataFrame is empty."], "warnings": [], "info": {}}

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Timestamp checks
    if "timestamp" in df.columns:
        bad_ts = df["timestamp"].isna().sum()
        if bad_ts > 0:
            warnings_list.append(f"{bad_ts} rows have invalid/unparseable timestamps.")
        if df["timestamp"].notna().sum() > 1:
            if not df["timestamp"].is_monotonic_increasing:
                warnings_list.append("Timestamps are not monotonically increasing — will be sorted.")

    # Numeric target check
    if "order_qty" in df.columns:
        null_qty = df["order_qty"].isna().sum()
        neg_qty  = (df["order_qty"] < 0).sum()
        if null_qty > 0:
            warnings_list.append(f"{null_qty} rows have missing order_qty (will be dropped).")
        if neg_qty > 0:
            warnings_list.append(f"{neg_qty} rows have negative order_qty.")

    # Duplicate detection
    if "timestamp" in df.columns and "series_id" in df.columns:
        dup_count = df.duplicated(subset=["series_id", "timestamp"]).sum()
        if dup_count > 0:
            warnings_list.append(f"{dup_count} duplicate (series_id, timestamp) rows detected — will be aggregated.")

    # Missing value summary
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    high_missing = missing_pct[missing_pct > 20]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            warnings_list.append(f"Column '{col}' has {pct}% missing values.")

    # Build info dict
    info = {}
    if not errors:
        info = get_dataset_summary(df)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "info": info,
    }


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary statistics dict for the dashboard info panel."""
    summary = {"n_rows": len(df), "n_cols": len(df.columns)}

    if "timestamp" in df.columns:
        valid_ts = df["timestamp"].dropna()
        if len(valid_ts):
            summary["date_min"] = str(valid_ts.min().date())
            summary["date_max"] = str(valid_ts.max().date())
            summary["date_range_days"] = (valid_ts.max() - valid_ts.min()).days

    if "series_id" in df.columns:
        summary["n_series"] = df["series_id"].nunique()
        summary["series_list"] = sorted(df["series_id"].unique().tolist())

    if "company_code" in df.columns:
        summary["companies"] = sorted(df["company_code"].unique().tolist())

    if "product_id" in df.columns:
        summary["products"] = sorted(df["product_id"].unique().tolist())

    for col in ["region", "city", "channel"]:
        if col in df.columns:
            summary[col + "s"] = sorted(df[col].dropna().unique().tolist())

    if "order_qty" in df.columns:
        summary["total_orders"] = int(df["order_qty"].sum())
        summary["avg_daily_orders"] = round(df["order_qty"].mean(), 1)
        summary["std_orders"] = round(df["order_qty"].std(), 1)

    summary["missing_pct"] = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
    summary["duplicate_rows"] = int(df.duplicated().sum())

    return summary


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_to_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate from order-level rows → one row per (series_id × timestamp).
    Covariates are averaged per timestamp; order_qty is summed.

    WHY: TimesFM & Chronos take a single time index per series.
    """
    grp_cols = ["series_id", "timestamp"]
    cov_cols = [
        "price", "promotion", "holiday", "is_weekend",
        "day_of_week", "month", "temperature_c", "rainfall_mm",
    ]
    if freq == "h":
        cov_cols += ["hour", "business_hour", "evening_peak"]

    agg_dict = {"order_qty": "sum"}
    for c in cov_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    # Preserve company/product for filtering downstream
    for meta_col in ["company_code", "product_id", "region", "city", "channel"]:
        if meta_col in df.columns and meta_col not in agg_dict:
            agg_dict[meta_col] = "first"

    agg = df.groupby(grp_cols, as_index=False).agg(agg_dict)
    agg = agg.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return agg


# ── Train/test split ──────────────────────────────────────────────────────────

def train_test_split_ts(series_df: pd.DataFrame, horizon: int):
    """
    Time-ordered split: last `horizon` rows → test, rest → train.
    Never shuffle a time series — future data would leak into training.
    """
    if len(series_df) <= horizon:
        raise ValueError(
            f"Series has {len(series_df)} rows but horizon={horizon}. "
            "Need at least horizon+1 rows."
        )
    return series_df.iloc[:-horizon].copy(), series_df.iloc[-horizon:].copy()


# ── Demo data generator ───────────────────────────────────────────────────────

def generate_demo_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic multi-product order data for demo mode.
    Returns an aggregated DataFrame ready for forecasting.
    """
    CONFIGS = {
        "KCI_P001": (150, 40, "KCI", "P001"),
        "KCI_P002": (122, 35, "KCI", "P002"),
        "KCI_P003": (54,  18, "KCI", "P003"),
        "KCI_P004": (118, 32, "KCI", "P004"),
        "MCI_P001": (179, 50, "MCI", "P001"),
        "MCI_P002": (146, 42, "MCI", "P002"),
        "MCI_P003": (63,  22, "MCI", "P003"),
        "MCI_P004": (138, 40, "MCI", "P004"),
    }
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for sid, (mu, sd, company, product) in CONFIGS.items():
        rng = np.random.default_rng(abs(hash(sid)) % (2**31))
        # Trend + seasonality + noise
        trend   = np.linspace(0, mu * 0.1, n_days)
        weekly  = np.sin(np.arange(n_days) * 2 * np.pi / 7) * sd * 0.25
        monthly = np.sin(np.arange(n_days) * 2 * np.pi / 30) * sd * 0.15
        noise   = rng.normal(0, sd * 0.4, n_days)
        vals    = np.clip(mu + trend + weekly + monthly + noise, 1, None).astype(int)

        for i, (d, v) in enumerate(zip(dates, vals)):
            dow = d.dayofweek
            rows.append({
                "series_id":    sid,
                "timestamp":    d,
                "order_qty":    v,
                "company_code": company,
                "product_id":   product,
                "price":        round(rng.uniform(30, 60), 2),
                "promotion":    int(rng.random() < 0.12),
                "holiday":      int(rng.random() < 0.03),
                "is_weekend":   int(dow >= 5),
                "day_of_week":  dow,
                "month":        d.month,
                "temperature_c": round(rng.uniform(16, 32), 1),
                "rainfall_mm":   round(max(0, rng.normal(5, 4)), 1),
            })

    df = pd.DataFrame(rows).sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return df
