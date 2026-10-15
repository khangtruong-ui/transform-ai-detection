"""Scatter plots analyzing image/mask diagnostic descriptors versus localization degradation."""

from __future__ import annotations

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_property_vs_degradation(
    metrics_csv_path: str,
    diagnostics_csv_path: str,
    output_dir: str,
    target_metric: str = "delta_iou",
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Optional[str]:
    """Plot scatter plots of pre-inference image properties against degradation."""
    if not os.path.exists(metrics_csv_path) or not os.path.exists(diagnostics_csv_path):
        return None

    df_m = pd.read_csv(metrics_csv_path)
    df_d = pd.read_csv(diagnostics_csv_path)

    # Use baseline identity diagnostics
    id_diag = df_d[df_d["transform_name"] == "identity"].drop_duplicates(subset=["sample_id"])
    if id_diag.empty:
        return None

    merged = pd.merge(df_m, id_diag, on="sample_id", how="inner", suffixes=("", "_diag"))
    # Filter for transformed conditions only (where degradation is meaningful)
    non_id = merged[merged["transform_name"] != "identity"]
    if non_id.empty or target_metric not in non_id.columns:
        return None

    features = [
        ("manipulation_area_fraction", "Manipulation Area Fraction"),
        ("isoperimetric_quotient", "Mask Compactness (4πA/P²)"),
        ("min_border_distance_norm", "Norm. Distance to Canvas Border"),
        ("grad_magnitude_mean", "Mean Sobel Gradient (Texture)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"Diagnostic Descriptors vs Degradation ({target_metric.upper()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    models = sorted(non_id["model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(models))))

    for idx, (feat, label) in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        if feat not in non_id.columns:
            continue

        valid = non_id.dropna(subset=[feat, target_metric])
        if valid.empty:
            continue

        for m_idx, m_name in enumerate(models):
            m_sub = valid[valid["model"] == m_name]
            if m_sub.empty:
                continue
            ax.scatter(
                m_sub[feat],
                m_sub[target_metric],
                alpha=0.4,
                color=colors[m_idx],
                label=m_name,
                edgecolors="none",
                s=25,
            )

        # Overall trendline
        x = valid[feat].values
        y = valid[target_metric].values
        if len(x) > 2 and np.std(x) > 1e-6:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.5, label="Overall Trend")

            sig_str = f"r = {r_val:.2f} (p={p_val:.3f})"
            ax.text(
                0.05,
                0.93,
                sig_str,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel(target_metric.upper(), fontsize=10)
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    base_path = os.path.join(output_dir, f"scatter_property_vs_{target_metric}")
    for fmt in export_formats:
        fig.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f"{base_path}.png"
