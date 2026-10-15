"""Model-by-transformation performance change heatmaps."""

from __future__ import annotations

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_performance_change_heatmap(
    aggregate_csv_path: str,
    output_dir: str,
    metric_delta: str = "delta_iou",
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Optional[str]:
    """Plot Model-by-Transformation degradation heatmap (relative to identity baseline)."""
    if not os.path.exists(aggregate_csv_path):
        return None

    df = pd.read_csv(aggregate_csv_path)
    col = f"{metric_delta}_mean"
    if df.empty or col not in df.columns:
        return None

    non_id = df[df["transform_name"] != "identity"].copy()
    if non_id.empty:
        return None

    # Condition label: transform_name (severity)
    non_id["condition"] = non_id["transform_name"] + " (" + non_id["severity_label"] + ")"

    pivot = non_id.pivot_table(
        index="model",
        columns="condition",
        values=col,
        aggfunc="mean",
    )

    if pivot.empty:
        return None

    plt.figure(figsize=(max(8, len(pivot.columns) * 1.1), max(4, len(pivot.index) * 0.9)))
    # Diverging colormap centered at 0
    vmin = min(-0.5, pivot.min().min())
    vmax = max(0.1, pivot.max().max())

    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Red (drop) to Green (gain)
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": f"Mean {metric_delta.upper()} relative to Identity"},
    )

    ax.set_title(
        f"Model Robustness Profile: {metric_delta.upper()} Across Transformations",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Localization Model", fontsize=11)
    ax.set_xlabel("Transformation Condition", fontsize=11)
    plt.xticks(rotation=40, ha="right")

    plt.tight_layout()
    base_path = os.path.join(output_dir, f"heatmap_{metric_delta}")
    for fmt in export_formats:
        plt.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    return f"{base_path}.png"
