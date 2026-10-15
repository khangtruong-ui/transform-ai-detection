"""Publication-quality performance versus transformation severity curves."""

from __future__ import annotations

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_performance_vs_severity(
    aggregate_csv_path: str,
    output_dir: str,
    metric: str = "iou",
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Optional[str]:
    """Plot localization performance versus transformation severity with 95% bootstrap CIs."""
    if not os.path.exists(aggregate_csv_path):
        return None

    df = pd.read_csv(aggregate_csv_path)
    if df.empty or f"{metric}_mean" not in df.columns:
        return None

    # Filter out identity baseline
    non_id = df[df["transform_name"] != "identity"]
    transforms = sorted(non_id["transform_name"].unique())
    models = sorted(df["model"].unique())

    if not transforms:
        return None

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    n_t = len(transforms)
    ncols = min(3, n_t)
    nrows = (n_t + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Localization Performance ({metric.upper()}) vs Transformation Severity", fontsize=14, y=1.02)

    for i, t_name in enumerate(transforms):
        row_idx = i // ncols
        col_idx = i % ncols
        ax = axes[row_idx, col_idx]

        sub = non_id[non_id["transform_name"] == t_name]
        for m_name in models:
            m_sub = sub[sub["model"] == m_name].sort_values("severity_label")
            if m_sub.empty:
                continue

            x_labels = m_sub["severity_label"].tolist()
            x = range(len(x_labels))
            y = m_sub[f"{metric}_mean"].values
            y_low = m_sub[f"{metric}_ci_lower"].values
            y_high = m_sub[f"{metric}_ci_upper"].values

            line = ax.plot(x, y, marker="o", label=m_name, linewidth=2)
            color = line[0].get_color()
            if not pd.isna(y_low).all() and not pd.isna(y_high).all():
                ax.fill_between(x, y_low, y_high, alpha=0.15, color=color)

            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=30, ha="right")

        ax.set_title(t_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Severity Level", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left")

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout()
    base_path = os.path.join(output_dir, f"perf_vs_severity_{metric}")
    for fmt in export_formats:
        fig.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f"{base_path}.png"
