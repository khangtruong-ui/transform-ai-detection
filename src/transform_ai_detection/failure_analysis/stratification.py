"""Performance stratification across image properties, generator methods, and datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from transform_ai_detection.statistics.bootstrap import compute_image_level_bootstrap_ci


def bin_mask_area(area_fraction: float) -> str:
    """Discretize ground-truth mask area fraction into interpretable tiers."""
    if area_fraction <= 1e-6:
        return "authentic (0%)"
    if area_fraction < 0.01:
        return "tiny (<1%)"
    if area_fraction < 0.05:
        return "small (1-5%)"
    if area_fraction < 0.20:
        return "medium (5-20%)"
    return "large (>20%)"


def bin_resolution(pixel_count: int) -> str:
    """Discretize image canvas resolution into pixel-count tiers."""
    if pixel_count < 250_000:  # e.g., 500x500
        return "low (<0.25 MP)"
    if pixel_count <= 1_000_000:  # up to 1000x1000
        return "medium (0.25-1 MP)"
    return "high (>1 MP)"


def stratify_performance(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: List[str] = ("iou", "f1", "delta_iou"),
    failure_col: str = "is_failure",
    n_bootstraps: int = 500,
) -> pd.DataFrame:
    """Stratify evaluation metrics across subgroup categories with bootstrap CIs."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    records = []
    for val, group in df.groupby(group_col):
        n_samples = len(group)
        rec: Dict[str, Any] = {
            group_col: val,
            "sample_count": n_samples,
        }

        if failure_col in group.columns:
            rec["failure_rate"] = float(group[failure_col].mean())

        for m_col in metric_cols:
            if m_col in group.columns:
                series = group[m_col].dropna().values
                boot = compute_image_level_bootstrap_ci(
                    series, n_bootstraps=n_bootstraps, ci=0.95
                )
                rec[f"{m_col}_mean"] = boot["mean"]
                rec[f"{m_col}_ci_lower"] = boot["ci_lower"]
                rec[f"{m_col}_ci_upper"] = boot["ci_upper"]

        records.append(rec)

    return pd.DataFrame(records)
