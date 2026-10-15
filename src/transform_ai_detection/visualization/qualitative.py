"""Representative qualitative visual cases generator."""

from __future__ import annotations

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_qualitative_sample(
    original_img: Image.Image,
    gt_mask: np.ndarray,
    transformed_img: Image.Image,
    pred_mask: np.ndarray,
    output_path: str,
    title_suffix: str = "",
    dpi: int = 300,
) -> str:
    """Plot a publication-grade 4-panel comparison of an image under evaluation."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    # 1. Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("Input Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Mask", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3. Transformed Image
    axes[2].imshow(transformed_img)
    axes[2].set_title("Transformed Image", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # 4. Error Overlay
    # Green = TP, Red = FP, Blue = FN
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(gt_mask > 0, pred_mask > 0)
    fp = np.logical_and(gt_mask == 0, pred_mask > 0)
    fn = np.logical_and(gt_mask > 0, pred_mask == 0)

    overlay[tp] = [46, 204, 113]  # Green (TP)
    overlay[fp] = [231, 76, 60]   # Red (FP)
    overlay[fn] = [52, 152, 219]  # Blue (FN)

    axes[3].imshow(overlay)
    axes[3].set_title("Error: TP(Grn), FP(Red), FN(Blu)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
