"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def synthetic_image_and_mask():
    """Generates a 128x128 synthetic image and ground truth binary mask."""
    w, h = 128, 128
    # Image with gradient and noise
    x = np.linspace(0, 2 * np.pi, w)
    y = np.linspace(0, 2 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    arr = (128 + 60 * np.sin(xx) * np.cos(yy)).astype(np.uint8)
    img_arr = np.repeat(arr[:, :, None], 3, axis=2)
    img = Image.fromarray(img_arr, mode="RGB")

    # Binary mask: circle in center
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.ellipse([40, 40, 88, 88], fill=255)
    mask = (np.array(mask_img) > 127).astype(np.uint8)

    return img, mask


@pytest.fixture
def empty_mask():
    """Generates a 128x128 all-zero binary mask for authentic image testing."""
    return np.zeros((128, 128), dtype=np.uint8)
