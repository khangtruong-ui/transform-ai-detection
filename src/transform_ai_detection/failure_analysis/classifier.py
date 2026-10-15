"""Continuous and thresholded failure classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class FailureClassification:
    is_failure: bool
    is_severe_degradation: bool
    is_transformation_induced_failure: bool
    failure_metric: str
    metric_value: float
    delta_value: Optional[float]


class FailureClassifier:
    """Classifies localization failure based on configurable continuous thresholds."""

    def __init__(
        self,
        failure_metric: str = "iou",
        failure_threshold: float = 0.3,
        degradation_delta_threshold: float = -0.2,
    ) -> None:
        self.failure_metric = failure_metric
        self.failure_threshold = failure_threshold
        self.degradation_delta_threshold = degradation_delta_threshold

    def classify(
        self,
        current_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> FailureClassification:
        val = current_metrics.get(self.failure_metric)
        if val is None or not np.isfinite(val):
            val = 0.0

        is_fail = bool(val < self.failure_threshold)

        delta_val = None
        is_severe_drop = False
        is_trans_induced = False

        if baseline_metrics is not None:
            base_val = baseline_metrics.get(self.failure_metric)
            if base_val is not None and np.isfinite(base_val):
                delta_val = float(val - base_val)
                is_severe_drop = bool(delta_val <= self.degradation_delta_threshold)
                # Transformation-induced: was passing on baseline, but failing now
                was_passing = bool(base_val >= self.failure_threshold)
                is_trans_induced = bool(was_passing and is_fail)

        return FailureClassification(
            is_failure=is_fail,
            is_severe_degradation=is_severe_drop,
            is_transformation_induced_failure=is_trans_induced,
            failure_metric=self.failure_metric,
            metric_value=float(val),
            delta_value=delta_val,
        )
