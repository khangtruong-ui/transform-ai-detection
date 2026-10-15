"""Subgroup performance plots across generator methods and datasets."""

from __future__ import annotations

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_subgroup_performance(
    stratification_csv_path: str,
    output_dir: str,
    group_col: str = "method",
    metric: str = "iou",
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Optional[str]:
    """Plot subgroup localization performance with bootstrap error bars."""
    if not os.path.exists(stratification_csv_path):
        return None

    df = pd.read_csv(stratification_csv_path)
    mean_col = f"{metric}_mean"
    if df.empty or mean_col not in df.columns or group_col not in df.columns:
        return None

    df_sorted = df.sort_values(mean_col, ascending=False)
    groups = df_sorted[group_col].tolist()
    means = df_sorted[mean_col].values
    ci_low = df_sorted[f"{metric}_ci_lower"].values
    ci_high = df_sorted[f"{metric}_ci_upper"].values

    err_low = np.maximum(0.0, means - ci_low)
    err_high = np.maximum(0.0, ci_high - means)
    yerr = [err_low, err_high]

    plt.figure(figsize=(max(8, len(groups) * 0.8), 5))
    bars = plt.bar(
        range(len(groups)),
        means,
        yerr=yerr,
        capsize=4,
        color="#3498db",
        edgecolor="black",
        linewidth=1.1,
    )

    plt.xticks(range(len(groups)), groups, rotation=35, ha="right", fontsize=9)
    plt.ylabel(f"Mean {metric.upper()}", fontsize=11)
    plt.xlabel(group_col.replace("_", " ").title(), fontsize=11)
    plt.title(
        f"Localization Performance ({metric.upper()}) Stratified by {group_col.replace('_', ' ').title()}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.ylim(-0.05, 1.05)

    # Annotate counts
    if "sample_count" in df_sorted.columns:
        counts = df_sorted["sample_count"].tolist()
        for i, (b, cnt) in enumerate(zip(bars, counts)):
            plt.text(
                b.get_x() + b.get_width() / 2.0,
                0.02,
                f"N={cnt}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()
    base_path = os.path.join(output_dir, f"subgroup_{group_col}_{metric}")
    for fmt in export_formats:
        plt.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    return f"{base_path}.png"
