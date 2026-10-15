"""Shared failure matrix and co-occurrence distributions."""

from __future__ import annotations

import json
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_shared_failure_analysis(
    failure_json_path: str,
    output_dir: str,
    export_formats: List[str] = ("png", "pdf"),
    dpi: int = 300,
) -> Optional[str]:
    """Plot pairwise model failure similarity and shared failure frequency."""
    if not os.path.exists(failure_json_path):
        return None

    with open(failure_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data or "pairwise_jaccard_similarity" not in data:
        return None

    jaccard_dict = data["pairwise_jaccard_similarity"]
    models = sorted(list(jaccard_dict.keys()))
    if len(models) < 2:
        return None

    df_jaccard = pd.DataFrame(jaccard_dict)[models].reindex(models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Model Failure Structure & Shared Vulnerabilities", fontsize=13, fontweight="bold")

    # 1. Pairwise Jaccard Similarity Heatmap
    sns.heatmap(
        df_jaccard,
        annot=True,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        ax=ax1,
        cbar_kws={"label": "Failure Jaccard Overlap"},
    )
    ax1.set_title("Pairwise Failure Overlap (Jaccard)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Model", fontsize=10)
    ax1.set_xlabel("Model", fontsize=10)

    # 2. Failure Category Breakdown
    categories = [
        "All Succeeded",
        "Model-Specific Failure",
        "Shared (All Failed)",
    ]
    counts = [
        data.get("no_models_failed_count", 0),
        data.get("model_specific_failure_count", 0),
        data.get("all_models_failed_count", 0),
    ]

    total = sum(counts)
    pcts = [100.0 * c / total if total > 0 else 0.0 for c in counts]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    bars = ax2.bar(categories, pcts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("Distribution of Failure Scope", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% of Evaluated Sample Instances", fontsize=10)
    ax2.set_ylim(0, max(100, max(pcts) * 1.15 if pcts else 100))

    for bar, pct, cnt in zip(bars, pcts, counts):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{pct:.1f}%\n(N={cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    base_path = os.path.join(output_dir, "shared_failure_analysis")
    for fmt in export_formats:
        fig.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f"{base_path}.png"
