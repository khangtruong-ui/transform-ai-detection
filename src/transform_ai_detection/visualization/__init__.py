"""Visualization package for publication-quality figures."""

from transform_ai_detection.visualization.curves import plot_performance_vs_severity
from transform_ai_detection.visualization.heatmaps import plot_performance_change_heatmap
from transform_ai_detection.visualization.scatter import plot_property_vs_degradation
from transform_ai_detection.visualization.shared_matrix import plot_shared_failure_analysis
from transform_ai_detection.visualization.subgroups import plot_subgroup_performance
from transform_ai_detection.visualization.qualitative import plot_qualitative_sample
from transform_ai_detection.visualization.plot_all import generate_all_plots

__all__ = [
    "plot_performance_vs_severity",
    "plot_performance_change_heatmap",
    "plot_property_vs_degradation",
    "plot_shared_failure_analysis",
    "plot_subgroup_performance",
    "plot_qualitative_sample",
    "generate_all_plots",
]
