"""Threshold-free ranking metrics: AUROC and Average Precision."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    max_pixels: int = 500_000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute pixel-level AUROC and Average Precision (AP / PR-AUC).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth mask {0, 1}.
    y_score : Optional[np.ndarray]
        Predicted continuous probability map or logit map in [0, 1] or unbounded.
        If None, ranking metrics cannot be computed.
    max_pixels : int
        Maximum pixel count for evaluation. If exceeded, pixels are sampled deterministically.
    seed : int
        Random seed for pixel subsampling.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (AUROC, Average Precision). Returns None for cases where metric is undefined
        (e.g., single-class ground truth like purely authentic images).
    """
    if y_score is None:
        return None, None

    y_true_flat = (y_true > 0).astype(np.int32).ravel()
    y_score_flat = np.asarray(y_score, dtype=np.float32).ravel()

    if y_true_flat.shape != y_score_flat.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    # Check for NaN / Inf in scores
    if not np.isfinite(y_score_flat).all():
        y_score_flat = np.nan_to_num(y_score_flat, nan=0.0, posinf=1.0, neginf=0.0)

    # If single-class (e.g. entirely 0s or entirely 1s), ROC AUC is undefined
    unique_true = np.unique(y_true_flat)
    if len(unique_true) < 2:
        return None, None

    # Deterministic subsampling if image exceeds max_pixels
    n_total = len(y_true_flat)
    if n_total > max_pixels:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=max_pixels, replace=False)
        y_true_sub = y_true_flat[idx]
        y_score_sub = y_score_flat[idx]
        if len(np.unique(y_true_sub)) < 2:
            # Fallback to stratified or entire array
            pos_idx = np.where(y_true_flat == 1)[0]
            neg_idx = np.where(y_true_flat == 0)[0]
            n_pos = min(len(pos_idx), max_pixels // 2)
            n_neg = min(len(neg_idx), max_pixels - n_pos)
            sample_idx = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ])
            y_true_sub = y_true_flat[sample_idx]
            y_score_sub = y_score_flat[sample_idx]
    else:
        y_true_sub = y_true_flat
        y_score_sub = y_score_flat

    try:
        auroc = float(roc_auc_score(y_true_sub, y_score_sub))
    except Exception as e:
        logger.debug(f"AUROC calculation failed: {e}")
        auroc = None

    try:
        ap = float(average_precision_score(y_true_sub, y_score_sub))
    except Exception as e:
        logger.debug(f"AP calculation failed: {e}")
        ap = None

    return auroc, ap
