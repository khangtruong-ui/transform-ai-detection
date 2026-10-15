"""Unit tests for transformations, geometry preservation, and metadata declarations."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transform_ai_detection.transformations.classical import (
    IdentityTransformation,
    JPEGCompressionTransformation,
    GaussianBlurTransformation,
    DownscaleUpscaleTransformation,
    BrightnessContrastTransformation,
)
from transform_ai_detection.transformations.geometric import (
    HorizontalFlipTransformation,
    VerticalFlipTransformation,
    CenterCropTransformation,
)
from transform_ai_detection.transformations.noise_sharp import (
    GaussianNoiseTransformation,
    SharpeningTransformation,
)


def test_identity_transformation(synthetic_image_and_mask):
    img, mask = synthetic_image_and_mask
    t = IdentityTransformation()
    res = t.apply(img, mask)

    assert res.is_valid_comparison
    assert res.image.size == img.size
    assert np.array_equal(res.mask, mask)
    meta = t.get_metadata()
    assert not meta.changes_geometry
    assert not meta.changes_dimensions


def test_classical_transformations_preserve_geometry_and_mask(synthetic_image_and_mask):
    img, mask = synthetic_image_and_mask
    transforms = [
        JPEGCompressionTransformation(quality=50),
        GaussianBlurTransformation(sigma=2.0),
        DownscaleUpscaleTransformation(scale_factor=0.5),
        BrightnessContrastTransformation(factor_brightness=1.2, factor_contrast=1.1),
        GaussianNoiseTransformation(std=0.03),
        SharpeningTransformation(factor=2.0),
    ]

    for t in transforms:
        res = t.apply(img, mask)
        meta = t.get_metadata()

        assert res.is_valid_comparison
        assert res.image.size == img.size
        # For non-geometric transformations, ground-truth mask must be strictly identical
        assert np.array_equal(res.mask, mask), f"Mask modified by non-geometric transform: {meta.name}"
        assert not meta.changes_geometry


def test_geometric_transformations_align_mask(synthetic_image_and_mask):
    img, mask = synthetic_image_and_mask

    # Horizontal flip
    hflip = HorizontalFlipTransformation()
    res_h = hflip.apply(img, mask)
    assert res_h.is_valid_comparison
    assert np.array_equal(res_h.mask, np.fliplr(mask))
    assert hflip.get_metadata().changes_geometry

    # Vertical flip
    vflip = VerticalFlipTransformation()
    res_v = vflip.apply(img, mask)
    assert res_v.is_valid_comparison
    assert np.array_equal(res_v.mask, np.flipud(mask))
    assert vflip.get_metadata().changes_geometry


def test_center_crop_invalidates_when_tampering_eliminated():
    # Create mask with tampering strictly in bottom-right corner
    w, h = 100, 100
    img = Image.new("RGB", (w, h), (128, 128, 128))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[90:100, 90:100] = 1  # only in corner

    # Center crop of ratio 0.5 will only take [25:75, 25:75], which excludes [90:100, 90:100]
    crop_t = CenterCropTransformation(crop_ratio=0.5)
    res = crop_t.apply(img, mask)

    assert not res.is_valid_comparison
    assert res.invalidation_reason == "TAMPERED_REGION_ELIMINATED_BY_CROP"
