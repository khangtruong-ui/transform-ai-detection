"""Unit tests for bootstrap confidence intervals, paired hypothesis tests, and correlations."""

from __future__ import annotations

import numpy as np
import pytest

from transform_ai_detection.statistics.bootstrap import (
    compute_image_level_bootstrap_ci,
    compute_paired_bootstrap_difference_ci,
)
from transform_ai_detection.statistics.paired_tests import run_paired_significance_test
from transform_ai_detection.statistics.correlation import (
    compute_bivariate_correlation,
    compute_partial_correlation,
)


def test_bootstrap_ci_coverage():
    # Synthetic normal sample with known mean ~ 0.5
    rng = np.random.RandomState(42)
    data = rng.normal(loc=0.5, scale=0.1, size=100)
    boot = compute_image_level_bootstrap_ci(data, n_bootstraps=1000, ci=0.95, seed=42)

    assert boot["mean"] is not None
    assert boot["ci_lower"] is not None
    assert boot["ci_upper"] is not None
    assert boot["ci_lower"] < boot["mean"] < boot["ci_upper"]
    assert 0.45 < boot["mean"] < 0.55


def test_bootstrap_ci_constant_array():
    # Array where all elements are identical
    data = np.full(50, 0.75)
    boot = compute_image_level_bootstrap_ci(data, n_bootstraps=500)
    assert boot["mean"] == 0.75
    assert boot["ci_lower"] == 0.75
    assert boot["ci_upper"] == 0.75
    assert boot["std_err"] == 0.0


def test_paired_bootstrap_difference():
    base = np.array([0.8, 0.85, 0.9, 0.75, 0.88])
    trans = np.array([0.5, 0.55, 0.6, 0.45, 0.58])  # dropped by ~0.3
    boot = compute_paired_bootstrap_difference_ci(base, trans, n_bootstraps=500)
    assert boot["mean"] < -0.25
    assert boot["ci_upper"] < 0.0


def test_paired_significance_tests():
    rng = np.random.RandomState(42)
    base = rng.uniform(0.7, 0.9, size=30)
    # Significant degradation
    trans = base - rng.uniform(0.1, 0.3, size=30)

    res = run_paired_significance_test(base, trans)
    assert res["n_samples"] == 30
    assert res["mean_difference"] < 0.0
    assert res["wilcoxon_pvalue"] is not None and res["wilcoxon_pvalue"] < 0.001
    assert res["is_significant_005"] is True
    assert res["cohens_d"] < -1.0


def test_bivariate_and_partial_correlation():
    rng = np.random.RandomState(42)
    # z is confounder
    z = rng.normal(0, 1, 100)
    # x and y both correlate with z
    x = 0.8 * z + rng.normal(0, 0.5, 100)
    y = 0.8 * z + rng.normal(0, 0.5, 100)

    # Raw correlation should be strongly positive
    raw = compute_bivariate_correlation(x, y)
    assert raw["pearson_r"] > 0.5

    # Partial correlation controlling for z should approach 0
    partial = compute_partial_correlation(x, y, z[:, None])
    assert abs(partial["pearson_r"]) < 0.25
