"""Inexpensive CPU-based texture and gradient statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, laplace


@dataclass
class TextureDiagnostics:
    grad_magnitude_mean: float
    grad_magnitude_std: float
    laplacian_variance: float
    intensity_contrast: float
    shannon_entropy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_texture_diagnostics(image: Image.Image) -> TextureDiagnostics:
    """Compute gradient, sharpness, and entropy statistics on image RGB/gray pixels."""
    # Convert to grayscale float in [0.0, 1.0]
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0

    # Sobel gradients
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    grad_mag = np.sqrt(sx**2 + sy**2)

    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # Laplacian variance (sharpness descriptor)
    lap = laplace(gray)
    lap_var = float(np.var(lap))

    # Intensity contrast (standard deviation of grayscale values)
    contrast = float(np.std(gray))

    # Shannon entropy
    counts, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    probs = counts / float(gray.size)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    return TextureDiagnostics(
        grad_magnitude_mean=grad_mean,
        grad_magnitude_std=grad_std,
        laplacian_variance=lap_var,
        intensity_contrast=contrast,
        shannon_entropy=entropy,
    )
