"""Diagnostics package for pre- and post-inference descriptor extraction."""

from transform_ai_detection.diagnostics.geometry import (
    GeometricDiagnostics,
    compute_geometric_diagnostics,
)
from transform_ai_detection.diagnostics.texture import (
    TextureDiagnostics,
    compute_texture_diagnostics,
)
from transform_ai_detection.diagnostics.error_diag import (
    PostInferenceDiagnostics,
    compute_post_inference_diagnostics,
)

__all__ = [
    "GeometricDiagnostics",
    "compute_geometric_diagnostics",
    "TextureDiagnostics",
    "compute_texture_diagnostics",
    "PostInferenceDiagnostics",
    "compute_post_inference_diagnostics",
]
