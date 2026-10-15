"""Unit tests for failure classification, shared failures, and stratification."""

from __future__ import annotations

import pandas as pd
import pytest

from transform_ai_detection.failure_analysis.classifier import FailureClassifier
from transform_ai_detection.failure_analysis.shared import analyze_shared_failures
from transform_ai_detection.failure_analysis.stratification import (
    bin_mask_area,
    bin_resolution,
    stratify_performance,
)


def test_failure_classifier():
    classifier = FailureClassifier(failure_threshold=0.3, degradation_delta_threshold=-0.2)

    # 1. Success on both
    c1 = classifier.classify({"iou": 0.7}, baseline_metrics={"iou": 0.8})
    assert not c1.is_failure
    assert not c1.is_severe_degradation
    assert not c1.is_transformation_induced_failure

    # 2. Transformation induced failure (was 0.5, dropped to 0.2)
    c2 = classifier.classify({"iou": 0.2}, baseline_metrics={"iou": 0.5})
    assert c2.is_failure
    assert c2.is_severe_degradation  # delta is -0.3 <= -0.2
    assert c2.is_transformation_induced_failure

    # 3. Baseline failure that remained failed (was 0.1, now 0.1)
    c3 = classifier.classify({"iou": 0.1}, baseline_metrics={"iou": 0.1})
    assert c3.is_failure
    assert not c3.is_transformation_induced_failure


def test_shared_failure_analysis():
    # Synthetic dataframe with 3 models evaluated on 3 samples
    records = [
        {"sample_id": "s1", "model": "m1", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": True},
        {"sample_id": "s1", "model": "m2", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": True},
        {"sample_id": "s2", "model": "m1", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": True},
        {"sample_id": "s2", "model": "m2", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": False},  # model-specific for m1
        {"sample_id": "s3", "model": "m1", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": False},
        {"sample_id": "s3", "model": "m2", "transform_name": "jpeg", "severity_label": "Q=50", "is_failure": False},  # both succeed
    ]
    df = pd.DataFrame(records)

    analysis = analyze_shared_failures(df, failure_col="is_failure")
    assert analysis["n_samples_complete"] == 3
    assert analysis["all_models_failed_count"] == 1  # s1
    assert analysis["no_models_failed_count"] == 1   # s3
    assert analysis["model_specific_failure_count"] == 1  # s2
    assert analysis["model_specific_breakdown"]["m1"] == 1
    assert analysis["model_specific_breakdown"]["m2"] == 0


def test_stratification():
    assert bin_mask_area(0.0) == "authentic (0%)"
    assert bin_mask_area(0.005) == "tiny (<1%)"
    assert bin_mask_area(0.03) == "small (1-5%)"
    assert bin_mask_area(0.15) == "medium (5-20%)"
    assert bin_mask_area(0.50) == "large (>20%)"

    df = pd.DataFrame([
        {"method": "Flux.1", "iou": 0.8, "is_failure": False},
        {"method": "Flux.1", "iou": 0.6, "is_failure": False},
        {"method": "SD1.5", "iou": 0.2, "is_failure": True},
    ])
    strat = stratify_performance(df, group_col="method", metric_cols=["iou"])
    assert len(strat) == 2
    flux_row = strat[strat["method"] == "Flux.1"].iloc[0]
    assert flux_row["sample_count"] == 2
    assert flux_row["failure_rate"] == 0.0
    assert flux_row["iou_mean"] == 0.7
