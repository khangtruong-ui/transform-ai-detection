"""Paired statistical hypothesis testing for degradation analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import stats


def run_paired_significance_test(
    base_values: Union[List[float], np.ndarray],
    trans_values: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """Perform paired hypothesis tests (Wilcoxon signed-rank and paired t-test).

    Parameters
    ----------
    base_values : array-like
        Baseline metric scores (e.g., IoU under identity).
    trans_values : array-like
        Transformed metric scores (e.g., IoU under JPEG compression).

    Returns
    -------
    Dict[str, Any]
        Test statistics, p-values, sample count, mean difference, Cohen's d.
    """
    base_arr = np.asarray(base_values, dtype=np.float64)
    trans_arr = np.asarray(trans_values, dtype=np.float64)

    valid = np.logical_and(np.isfinite(base_arr), np.isfinite(trans_arr))
    b = base_arr[valid]
    t = trans_arr[valid]
    n = len(b)

    if n < 2:
        return {
            "n_samples": n,
            "mean_difference": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "ttest_stat": None,
            "ttest_pvalue": None,
            "cohens_d": None,
            "is_significant_005": False,
        }

    deltas = t - b
    mean_diff = float(np.mean(deltas))
    std_diff = float(np.std(deltas, ddof=1))

    # Cohen's d for paired samples
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-9 else 0.0

    # Paired Student t-test
    try:
        t_res = stats.ttest_rel(t, b)
        ttest_stat = float(t_res.statistic)
        ttest_pvalue = float(t_res.pvalue)
    except Exception:
        ttest_stat, ttest_pvalue = None, None

    # Wilcoxon signed-rank test
    # If all deltas are 0, wilcoxon cannot run; p-value is 1.0
    if np.all(np.abs(deltas) < 1e-9):
        wilcoxon_stat = 0.0
        wilcoxon_pvalue = 1.0
    else:
        try:
            w_res = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_pvalue = float(w_res.pvalue)
        except Exception:
            wilcoxon_stat, wilcoxon_pvalue = None, None

    is_sig = bool(wilcoxon_pvalue is not None and wilcoxon_pvalue < 0.05)

    return {
        "n_samples": n,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "ttest_stat": ttest_stat,
        "ttest_pvalue": ttest_pvalue,
        "cohens_d": cohens_d,
        "is_significant_005": is_sig,
    }
