"""Failure analysis package."""

from transform_ai_detection.failure_analysis.classifier import (
    FailureClassification,
    FailureClassifier,
)
from transform_ai_detection.failure_analysis.shared import analyze_shared_failures
from transform_ai_detection.failure_analysis.stratification import (
    bin_mask_area,
    bin_resolution,
    stratify_performance,
)

__all__ = [
    "FailureClassification",
    "FailureClassifier",
    "analyze_shared_failures",
    "bin_mask_area",
    "bin_resolution",
    "stratify_performance",
]
