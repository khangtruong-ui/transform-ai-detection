"""Consolidated plot generator executing all publication visualizations from disk."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from transform_ai_detection.visualization.curves import plot_performance_vs_severity
from transform_ai_detection.visualization.heatmaps import plot_performance_change_heatmap
from transform_ai_detection.visualization.scatter import plot_property_vs_degradation
from transform_ai_detection.visualization.shared_matrix import plot_shared_failure_analysis
from transform_ai_detection.visualization.subgroups import plot_subgroup_performance

logger = logging.getLogger(__name__)


def generate_all_plots(
    output_dir: str,
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Dict[str, Optional[str]]:
    """Generate all standard publication-quality plots from saved results without running inference."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    agg_csv = os.path.join(output_dir, "aggregate_metrics.csv")
    metrics_csv = os.path.join(output_dir, "metrics_per_image.csv")
    diag_csv = os.path.join(output_dir, "diagnostics_per_image.csv")
    fail_json = os.path.join(output_dir, "failure_analysis.json")
    strat_method_csv = os.path.join(output_dir, "stratification_by_method.csv")
    strat_area_csv = os.path.join(output_dir, "stratification_by_mask_area.csv")

    generated = {}

    # 1. Performance vs Severity curves
    p1 = plot_performance_vs_severity(agg_csv, plots_dir, metric="iou", export_formats=export_formats, dpi=dpi)
    p2 = plot_performance_vs_severity(agg_csv, plots_dir, metric="f1", export_formats=export_formats, dpi=dpi)
    generated["perf_vs_severity_iou"] = p1
    generated["perf_vs_severity_f1"] = p2

    # 2. Performance Change Heatmaps
    p3 = plot_performance_change_heatmap(agg_csv, plots_dir, metric_delta="delta_iou", export_formats=export_formats, dpi=dpi)
    p4 = plot_performance_change_heatmap(agg_csv, plots_dir, metric_delta="delta_f1", export_formats=export_formats, dpi=dpi)
    generated["heatmap_delta_iou"] = p3
    generated["heatmap_delta_f1"] = p4

    # 3. Property vs Degradation Scatters
    p5 = plot_property_vs_degradation(metrics_csv, diag_csv, plots_dir, target_metric="delta_iou", export_formats=export_formats, dpi=dpi)
    generated["scatter_property_vs_degradation"] = p5

    # 4. Shared Failure Analysis
    p6 = plot_shared_failure_analysis(fail_json, plots_dir, export_formats=export_formats, dpi=dpi)
    generated["shared_failure_analysis"] = p6

    # 5. Subgroup Plots
    p7 = plot_subgroup_performance(strat_method_csv, plots_dir, group_col="method", metric="iou", export_formats=export_formats, dpi=dpi)
    p8 = plot_subgroup_performance(strat_area_csv, plots_dir, group_col="mask_area_tier", metric="iou", export_formats=export_formats, dpi=dpi)
    generated["subgroup_method"] = p7
    generated["subgroup_mask_area"] = p8

    logger.info(f"Generated {len([v for v in generated.values() if v is not None])} figures in {plots_dir}")
    return generated
