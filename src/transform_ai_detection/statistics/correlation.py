"""Correlation and regression analysis with confounder controls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_bivariate_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute Pearson and Spearman rank correlation between two vectors."""
    valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    xv = x[valid]
    yv = y[valid]

    if len(xv) < 3 or np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "n": len(xv),
        }

    try:
        pr, pp = stats.pearsonr(xv, yv)
    except Exception:
        pr, pp = None, None

    try:
        sr, sp = stats.spearmanr(xv, yv)
    except Exception:
        sr, sp = None, None

    return {
        "pearson_r": float(pr) if pr is not None else None,
        "pearson_p": float(pp) if pp is not None else None,
        "spearman_rho": float(sr) if sr is not None else None,
        "spearman_p": float(sp) if sp is not None else None,
        "n": len(xv),
    }


def compute_partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute partial correlation between x and y controlling for linear effects of covariates.

    Uses OLS residual orthogonalization (Frisch-Waugh-Lovell theorem).
    """
    valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    valid = np.logical_and(valid, np.all(np.isfinite(covariates), axis=1))

    xv = x[valid]
    yv = y[valid]
    zv = covariates[valid]

    n = len(xv)
    p = zv.shape[1]

    if n <= p + 2 or np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return {"partial_r": None, "p_value": None, "n": n}

    # Add intercept column
    Z = np.column_stack([np.ones(n), zv])

    # Regress x on Z
    beta_x, _, _, _ = np.linalg.lstsq(Z, xv, rcond=None)
    res_x = xv - Z @ beta_x

    # Regress y on Z
    beta_y, _, _, _ = np.linalg.lstsq(Z, yv, rcond=None)
    res_y = yv - Z @ beta_y

    return compute_bivariate_correlation(res_x, res_y)
