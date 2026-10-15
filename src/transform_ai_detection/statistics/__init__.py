"""Statistical analysis package."""

from transform_ai_detection.statistics.bootstrap import (
    compute_image_level_bootstrap_ci,
    compute_paired_bootstrap_difference_ci,
)
from transform_ai_detection.statistics.paired_tests import run_paired_significance_test
from transform_ai_detection.statistics.correlation import (
    compute_bivariate_correlation,
    compute_partial_correlation,
)

__all__ = [
    "compute_image_level_bootstrap_ci",
    "compute_paired_bootstrap_difference_ci",
    "run_paired_significance_test",
    "compute_bivariate_correlation",
    "compute_partial_correlation",
]
