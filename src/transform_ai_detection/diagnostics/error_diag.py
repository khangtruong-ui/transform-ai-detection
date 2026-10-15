"""Post-inference model-dependent error diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np
from PIL import Image
from scipy.ndimage import (
    label,
    generate_binary_structure,
    distance_transform_edt,
    sobel,
)

from transform_ai_detection.metrics.boundary import extract_boundary


@dataclass
class PostInferenceDiagnostics:
    false_positive_area_fraction: float
    false_negative_area_fraction: float
    prediction_fragmentation: int  # CC_pred - CC_gt
    boundary_displacement_mean: float  # Mean distance from FP/FN boundary to true boundary
    semantic_boundary_attraction_ratio: float  # Ratio of edge gradient at error pixels vs background

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_post_inference_diagnostics(
    image: Image.Image,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PostInferenceDiagnostics:
    """Compute post-inference failure and error descriptors.

    Note on semantic-boundary attraction:
    This metric measures whether false positive boundaries correlate with high image
    gradients (natural semantic edges). It is a diagnostic hypothesis to test, not
    conclusive proof of a particular failure mechanism.
    """
    total_pixels = y_true.size
    y_true_b = (y_true > 0).astype(bool)
    y_pred_b = (y_pred > 0).astype(bool)

    # 1. FP and FN areas
    fp_mask = np.logical_and(~y_true_b, y_pred_b)
    fn_mask = np.logical_and(y_true_b, ~y_pred_b)

    fp_area_frac = float(fp_mask.sum() / total_pixels) if total_pixels > 0 else 0.0
    fn_area_frac = float(fn_mask.sum() / total_pixels) if total_pixels > 0 else 0.0

    # 2. Fragmentation: difference in connected components count
    struct = generate_binary_structure(2, 2)
    _, num_gt_cc = label(y_true_b, structure=struct)
    _, num_pred_cc = label(y_pred_b, structure=struct)
    fragmentation = int(num_pred_cc - num_gt_cc)

    # 3. Boundary displacement: average distance of error boundary pixels to true boundary
    gt_boundary = extract_boundary(y_true)
    pred_boundary = extract_boundary(y_pred)

    if gt_boundary.sum() > 0 and pred_boundary.sum() > 0:
        d_to_gt = distance_transform_edt(~gt_boundary)
        err_boundary = np.logical_xor(gt_boundary, pred_boundary)
        if err_boundary.sum() > 0:
            displacement = float(np.mean(d_to_gt[err_boundary]))
        else:
            displacement = 0.0
    else:
        displacement = 0.0

    # 4. Semantic boundary attraction:
    # Compute image gradient magnitude
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    grad_mag = np.sqrt(sx**2 + sy**2)

    bg_mask = ~y_true_b
    mean_bg_grad = float(np.mean(grad_mag[bg_mask])) if bg_mask.sum() > 0 else 1.0

    if fp_mask.sum() > 0 and mean_bg_grad > 1e-6:
        mean_fp_grad = float(np.mean(grad_mag[fp_mask]))
        attraction_ratio = float(mean_fp_grad / mean_bg_grad)
    else:
        attraction_ratio = 1.0

    return PostInferenceDiagnostics(
        false_positive_area_fraction=fp_area_frac,
        false_negative_area_fraction=fn_area_frac,
        prediction_fragmentation=fragmentation,
        boundary_displacement_mean=displacement,
        semantic_boundary_attraction_ratio=attraction_ratio,
    )
