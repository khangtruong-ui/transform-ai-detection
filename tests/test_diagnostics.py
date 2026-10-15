"""Unit tests for pre- and post-inference image/mask diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transform_ai_detection.diagnostics.geometry import compute_geometric_diagnostics
from transform_ai_detection.diagnostics.texture import compute_texture_diagnostics
from transform_ai_detection.diagnostics.error_diag import compute_post_inference_diagnostics


def test_geometric_diagnostics_circle(synthetic_image_and_mask):
    img, mask = synthetic_image_and_mask
    diag = compute_geometric_diagnostics(img.width, img.height, mask)

    assert diag.width == 128
    assert diag.height == 128
    assert diag.pixel_count == 128 * 128
    assert 0.05 < diag.manipulation_area_fraction < 0.25
    assert diag.connected_components_count == 1
    # Circular mask has high isoperimetric quotient (> 0.5)
    assert diag.isoperimetric_quotient > 0.5
    assert 0.4 < diag.centroid_x_norm < 0.6
    assert 0.4 < diag.centroid_y_norm < 0.6


def test_geometric_diagnostics_empty_mask(empty_mask):
    diag = compute_geometric_diagnostics(128, 128, empty_mask)
    assert diag.manipulation_area_fraction == 0.0
    assert diag.connected_components_count == 0
    assert diag.perimeter == 0.0


def test_texture_diagnostics(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    tex = compute_texture_diagnostics(img)

    assert tex.grad_magnitude_mean > 0.0
    assert tex.laplacian_variance > 0.0
    assert tex.intensity_contrast > 0.0
    assert tex.shannon_entropy > 0.0


def test_post_inference_diagnostics():
    w, h = 64, 64
    img = Image.new("RGB", (w, h), (128, 128, 128))
    y_true = np.zeros((h, w), dtype=np.uint8)
    y_true[20:40, 20:40] = 1

    y_pred = np.zeros((h, w), dtype=np.uint8)
    y_pred[25:45, 25:45] = 1

    post = compute_post_inference_diagnostics(img, y_true, y_pred)
    assert post.false_positive_area_fraction > 0.0
    assert post.false_negative_area_fraction > 0.0
    assert post.prediction_fragmentation == 0  # 1 CC in both
