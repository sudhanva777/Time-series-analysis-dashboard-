"""
forecasting/covariates.py
─────────────────────────
Covariate matrix building and scaling logic.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_covariate_matrix(
    series_df,
    cov_cols: list,
    scaler: StandardScaler = None,
    fit: bool = True,
) -> tuple:
    """
    Extract and scale the covariate matrix from a series DataFrame.

    WHY SCALE: Foundation models are sensitive to covariate magnitude.
    Standardising keeps gradients stable and prevents price (≈40)
    dominating rainfall (≈5).

    Parameters
    ----------
    series_df : DataFrame with covariate columns
    cov_cols  : list of column names to use as covariates
    scaler    : existing StandardScaler (pass when fit=False)
    fit       : if True, fit a new scaler; if False, use existing scaler

    Returns
    -------
    (X, scaler) where X is float32 array of shape [T, len(cov_cols)]
    """
    # Only keep columns that exist in the DataFrame
    avail = [c for c in cov_cols if c in series_df.columns]
    if not avail:
        # Return zero matrix if no covariates available
        n = len(series_df)
        return np.zeros((n, 1), dtype=np.float32), StandardScaler()

    X = series_df[avail].ffill().fillna(0).values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, scaler
