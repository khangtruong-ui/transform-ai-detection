"""Evaluation engine integrating pixel, ranking, and boundary metrics with delta tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

from transform_ai_detection.metrics.pixel import compute_pixel_metrics
from transform_ai_detection.metrics.ranking import compute_ranking_metrics
from transform_ai_detection.metrics.boundary import compute_boundary_metrics


class EvaluationEngine:
    """Computes full metric suite for localization predictions."""

    def __init__(
        self,
        boundary_tolerance: int = 3,
        compute_ranking: bool = True,
        max_ranking_pixels: int = 500_000,
    ) -> None:
        self.boundary_tolerance = boundary_tolerance
        self.compute_ranking = compute_ranking
        self.max_ranking_pixels = max_ranking_pixels

    def evaluate_sample(
        self,
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
        y_pred_prob: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Compute all pixel, ranking, and boundary metrics for a single prediction."""
        pixel_res = compute_pixel_metrics(y_true, y_pred_binary)
        boundary_res = compute_boundary_metrics(
            y_true, y_pred_binary, tolerance=self.boundary_tolerance
        )

        auroc, ap = (None, None)
        if self.compute_ranking and y_pred_prob is not None:
            auroc, ap = compute_ranking_metrics(
                y_true, y_pred_prob, max_pixels=self.max_ranking_pixels, seed=seed
            )

        results: Dict[str, Any] = {
            "precision": pixel_res.precision,
            "recall": pixel_res.recall,
            "f1": pixel_res.f1,
            "iou": pixel_res.iou,
            "fpr": pixel_res.fpr,
            "fnr": pixel_res.fnr,
            "gt_area_fraction": pixel_res.gt_area_fraction,
            "pred_area_fraction": pixel_res.pred_area_fraction,
            "tp": pixel_res.tp,
            "fp": pixel_res.fp,
            "tn": pixel_res.tn,
            "fn": pixel_res.fn,
            "boundary_precision": boundary_res.boundary_precision,
            "boundary_recall": boundary_res.boundary_recall,
            "boundary_f1": boundary_res.boundary_f1,
            "auroc": auroc,
            "average_precision": ap,
        }
        return results

    @staticmethod
    def compute_deltas(
        baseline_metrics: Dict[str, Any],
        transformed_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute performance change relative to the identity baseline (transformed - baseline)."""
        deltas: Dict[str, Any] = {}
        for k, base_val in baseline_metrics.items():
            if k in ("tp", "fp", "tn", "fn"):
                continue
            trans_val = transformed_metrics.get(k)
            if base_val is not None and trans_val is not None:
                try:
                    deltas[f"delta_{k}"] = float(trans_val - base_val)
                except (TypeError, ValueError):
                    deltas[f"delta_{k}"] = None
            else:
                deltas[f"delta_{k}"] = None
        return deltas
