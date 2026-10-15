"""Image-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_image_level_bootstrap_ci(
    values: Union[List[float], np.ndarray],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute image-level non-parametric bootstrap confidence interval.

    Crucially, resampling occurs across image units (not pixel units) to preserve
    intra-image pixel spatial correlations and satisfy independence assumptions.

    Parameters
    ----------
    values : List[float] or np.ndarray
        Metric values across distinct images.
    n_bootstraps : int
        Number of bootstrap replicates.
    ci : float
        Confidence level in (0, 1), default 0.95.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, Optional[float]]
        {"mean": float, "ci_lower": float, "ci_upper": float, "std_err": float}
    """
    arr = np.asarray(values, dtype=np.float64)
    # Filter out NaNs
    arr = arr[np.isfinite(arr)]

    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "std_err": None}
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std_err": 0.0}

    # If all values are identical
    if np.all(arr == arr[0]):
        v = float(arr[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std_err": 0.0}

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    std_err = float(np.std(boot_means))
    mean_val = float(np.mean(arr))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
    }


def compute_paired_bootstrap_difference_ci(
    base_values: Union[List[float], np.ndarray],
    trans_values: Union[List[float], np.ndarray],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute bootstrap CI for paired difference Delta = trans - base across images."""
    base_arr = np.asarray(base_values, dtype=np.float64)
    trans_arr = np.asarray(trans_values, dtype=np.float64)

    if len(base_arr) != len(trans_arr):
        raise ValueError(f"Length mismatch: {len(base_arr)} vs {len(trans_arr)}")

    # Keep only pairs where both are finite
    valid = np.logical_and(np.isfinite(base_arr), np.isfinite(trans_arr))
    deltas = trans_arr[valid] - base_arr[valid]

    return compute_image_level_bootstrap_ci(
        deltas, n_bootstraps=n_bootstraps, ci=ci, seed=seed
    )
