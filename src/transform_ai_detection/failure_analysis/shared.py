"""Shared versus model-specific failure analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def analyze_shared_failures(
    results_df: pd.DataFrame,
    failure_col: str = "is_failure",
) -> Dict[str, Any]:
    """Analyze shared and model-specific failures across the model set.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing columns: sample_id, model, transform_name, severity_label, and failure_col.

    Returns
    -------
    Dict[str, Any]
        Summary containing failure overlap matrices, model-specific failure rates,
        and transformation-level correlation shifts.
    """
    if results_df.empty or "model" not in results_df.columns:
        return {}

    models = sorted(results_df["model"].unique().tolist())
    n_models = len(models)

    # Pivot to sample x model matrix for each transform condition
    pivoted = results_df.pivot_table(
        index=["sample_id", "transform_name", "severity_label"],
        columns="model",
        values=failure_col,
        aggfunc="first",
    ).dropna()

    if pivoted.empty:
        return {"models": models, "n_samples_complete": 0}

    # Total samples evaluated across all models
    n_complete = len(pivoted)

    # Count how many models failed for each sample
    fail_counts = pivoted.sum(axis=1)

    all_fail_count = int((fail_counts == n_models).sum())
    no_fail_count = int((fail_counts == 0).sum())
    model_specific_count = int((fail_counts == 1).sum())

    # Model-specific failure breakdown (failing ONLY on model m)
    model_specific_breakdown: Dict[str, int] = {}
    for m in models:
        only_m = (pivoted[m] == 1) & (pivoted.drop(columns=[m]).sum(axis=1) == 0)
        model_specific_breakdown[m] = int(only_m.sum())

    # Pairwise failure Jaccard similarity and correlation matrix
    jaccard_matrix: Dict[str, Dict[str, float]] = {m1: {} for m1 in models}
    corr_matrix: Dict[str, Dict[str, float]] = {m1: {} for m1 in models}

    for m1 in models:
        for m2 in models:
            s1 = pivoted[m1].astype(bool)
            s2 = pivoted[m2].astype(bool)

            union_count = (s1 | s2).sum()
            intersection_count = (s1 & s2).sum()

            jaccard = float(intersection_count / union_count) if union_count > 0 else 1.0
            jaccard_matrix[m1][m2] = jaccard

            # Correlation
            if s1.std() > 0 and s2.std() > 0:
                corr = float(np.corrcoef(s1.astype(float), s2.astype(float))[0, 1])
            else:
                corr = 1.0 if (s1 == s2).all() else 0.0
            corr_matrix[m1][m2] = corr

    return {
        "models": models,
        "n_samples_complete": n_complete,
        "all_models_failed_count": all_fail_count,
        "all_models_failed_rate": float(all_fail_count / n_complete) if n_complete > 0 else 0.0,
        "no_models_failed_count": no_fail_count,
        "no_models_failed_rate": float(no_fail_count / n_complete) if n_complete > 0 else 0.0,
        "model_specific_failure_count": model_specific_count,
        "model_specific_breakdown": model_specific_breakdown,
        "pairwise_jaccard_similarity": jaccard_matrix,
        "pairwise_correlation": corr_matrix,
    }
