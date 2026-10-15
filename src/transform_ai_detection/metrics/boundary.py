"""Boundary F-score (BF-score) metric with configurable tolerance distance."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure


@dataclass
class BoundaryMetrics:
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    tolerance: int

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        return {k: float(v) for k, v in d.items()}


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel wide morphological inner boundary of a binary mask."""
    b_mask = (mask > 0).astype(bool)
    if not b_mask.any():
        return np.zeros_like(b_mask)

    struct = generate_binary_structure(2, 2)  # 8-connectivity
    dilated = binary_dilation(b_mask, structure=struct)
    boundary = np.logical_xor(dilated, b_mask)
    return boundary


def compute_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 3,
) -> BoundaryMetrics:
    """Compute Boundary Precision, Recall, and F1 at specified pixel tolerance.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth mask.
    y_pred : np.ndarray
        Binary predicted mask.
    tolerance : int
        Distance threshold in pixels (theta) within which boundary alignment is credited.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    gt_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    n_gt = int(gt_boundary.sum())
    n_pred = int(pred_boundary.sum())

    # Case 1: Neither mask has any boundary (e.g. authentic image and all-black pred)
    if n_gt == 0 and n_pred == 0:
        return BoundaryMetrics(
            boundary_precision=1.0,
            boundary_recall=1.0,
            boundary_f1=1.0,
            tolerance=tolerance,
        )

    # Case 2: One has boundary, other does not
    if n_gt == 0 or n_pred == 0:
        return BoundaryMetrics(
            boundary_precision=0.0,
            boundary_recall=0.0,
            boundary_f1=0.0,
            tolerance=tolerance,
        )

    # Distance transforms
    # distance to gt boundary
    d_gt = distance_transform_edt(~gt_boundary)
    # distance to pred boundary
    d_pred = distance_transform_edt(~pred_boundary)

    # Precision: fraction of pred boundary pixels within tolerance of GT boundary
    pred_within_tol = np.logical_and(pred_boundary, d_gt <= tolerance).sum()
    precision = float(pred_within_tol / n_pred)

    # Recall: fraction of GT boundary pixels within tolerance of pred boundary
    gt_within_tol = np.logical_and(gt_boundary, d_pred <= tolerance).sum()
    recall = float(gt_within_tol / n_gt)

    denom = precision + recall
    f1 = float((2 * precision * recall) / denom) if denom > 0 else 0.0

    return BoundaryMetrics(
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        tolerance=tolerance,
    )
