"""Pre-inference geometric and morphological image/mask descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial import ConvexHull

from transform_ai_detection.metrics.boundary import extract_boundary


@dataclass
class GeometricDiagnostics:
    width: int
    height: int
    aspect_ratio: float
    pixel_count: int
    manipulation_area_fraction: float
    connected_components_count: int
    perimeter: float
    perimeter_to_area_ratio: float
    isoperimetric_quotient: float  # 4 * pi * Area / Perimeter^2
    convex_solidity: float  # Area / ConvexHullArea
    centroid_x_norm: float
    centroid_y_norm: float
    min_border_distance_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_geometric_diagnostics(
    width: int,
    height: int,
    mask: np.ndarray,
) -> GeometricDiagnostics:
    """Extract deterministic, CPU-based morphological and geometric descriptors.

    All descriptors are strictly computed before model inference.
    """
    aspect_ratio = float(width / height) if height > 0 else 1.0
    pixel_count = int(width * height)

    b_mask = (mask > 0).astype(bool)
    tampered_pixels = int(b_mask.sum())
    area_fraction = float(tampered_pixels / pixel_count) if pixel_count > 0 else 0.0

    if tampered_pixels == 0:
        # Authentic image or empty mask
        return GeometricDiagnostics(
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            pixel_count=pixel_count,
            manipulation_area_fraction=0.0,
            connected_components_count=0,
            perimeter=0.0,
            perimeter_to_area_ratio=0.0,
            isoperimetric_quotient=0.0,
            convex_solidity=1.0,
            centroid_x_norm=0.5,
            centroid_y_norm=0.5,
            min_border_distance_norm=1.0,
        )

    # 1. Connected components
    struct = generate_binary_structure(2, 2)  # 8-connectivity
    labeled_mask, num_features = label(b_mask, structure=struct)

    # 2. Perimeter (boundary pixel count)
    boundary = extract_boundary(mask)
    perimeter = float(boundary.sum())
    p_to_a = float(perimeter / tampered_pixels) if tampered_pixels > 0 else 0.0

    # 3. Isoperimetric quotient (compactness)
    # Circle = 1.0, jagged/irregular shapes << 1.0
    if perimeter > 0:
        iso_quotient = float(4.0 * math.pi * tampered_pixels / (perimeter**2))
    else:
        iso_quotient = 0.0

    # 4. Convex solidity (Area / Convex Hull Area)
    coords = np.argwhere(b_mask)  # rows (y), cols (x)
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            hull_area = float(hull.volume)  # In 2D, hull.volume is the 2D area
            solidity = float(tampered_pixels / hull_area) if hull_area > 0 else 1.0
            solidity = min(1.0, max(0.0, solidity))
        except Exception:
            solidity = 1.0
    else:
        solidity = 1.0

    # 5. Centroid
    cy, cx = coords.mean(axis=0)
    centroid_x_norm = float(cx / width)
    centroid_y_norm = float(cy / height)

    # 6. Minimum distance to image borders
    min_y = coords[:, 0].min()
    max_y = coords[:, 0].max()
    min_x = coords[:, 1].min()
    max_x = coords[:, 1].max()

    dist_top = min_y
    dist_bottom = height - 1 - max_y
    dist_left = min_x
    dist_right = width - 1 - max_x
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    norm_factor = min(width, height)
    min_border_distance_norm = float(min_dist / norm_factor) if norm_factor > 0 else 0.0

    return GeometricDiagnostics(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        pixel_count=pixel_count,
        manipulation_area_fraction=area_fraction,
        connected_components_count=int(num_features),
        perimeter=perimeter,
        perimeter_to_area_ratio=p_to_a,
        isoperimetric_quotient=iso_quotient,
        convex_solidity=solidity,
        centroid_x_norm=centroid_x_norm,
        centroid_y_norm=centroid_y_norm,
        min_border_distance_norm=min_border_distance_norm,
    )
