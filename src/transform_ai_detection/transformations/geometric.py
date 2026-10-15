"""Geometric transformations with consistent ground-truth mask alignment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from transform_ai_detection.transformations.base import (
    BaseTransformation,
    TransformMetadata,
    TransformResult,
    TransformationRegistry,
)


@TransformationRegistry.register("horizontal_flip")
class HorizontalFlipTransformation(BaseTransformation):
    """Horizontal mirror reflection applied consistently to image and mask."""

    def __init__(self, severity_level: int = 1) -> None:
        self.severity_level = severity_level

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        flipped_img = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_mask = np.fliplr(mask)

        return TransformResult(
            image=flipped_img,
            mask=flipped_mask,
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="horizontal_flip",
            severity_level=self.severity_level,
            severity_label="hflip",
            changes_geometry=True,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="nearest",
            parameters={},
            limitations=(
                "Reflects spatial positions; tests directional or positional "
                "biases in detector representations."
            ),
        )


@TransformationRegistry.register("vertical_flip")
class VerticalFlipTransformation(BaseTransformation):
    """Vertical mirror reflection applied consistently to image and mask."""

    def __init__(self, severity_level: int = 1) -> None:
        self.severity_level = severity_level

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        flipped_img = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        flipped_mask = np.flipud(mask)

        return TransformResult(
            image=flipped_img,
            mask=flipped_mask,
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="vertical_flip",
            severity_level=self.severity_level,
            severity_label="vflip",
            changes_geometry=True,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="nearest",
            parameters={},
            limitations="Inverts vertical orientation; gravity and lighting prior violations.",
        )


@TransformationRegistry.register("center_crop")
class CenterCropTransformation(BaseTransformation):
    """Center crop transformation with boundary check for manipulated regions."""

    def __init__(self, crop_ratio: float = 0.8) -> None:
        self.crop_ratio = float(crop_ratio)
        if not (0.1 <= self.crop_ratio < 1.0):
            raise ValueError(f"crop_ratio must be in [0.1, 1.0), got {self.crop_ratio}")

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        w, h = image.size
        crop_w = int(round(w * self.crop_ratio))
        crop_h = int(round(h * self.crop_ratio))

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        cropped_img = image.crop((left, top, right, bottom))
        cropped_mask = mask[top:bottom, left:right]

        # Invalidation check: if the original mask had tampered pixels, but the crop eliminated them
        orig_had_tampering = bool(mask.sum() > 0)
        crop_has_tampering = bool(cropped_mask.sum() > 0)

        is_valid = True
        invalidation_reason = None
        if orig_had_tampering and not crop_has_tampering:
            is_valid = False
            invalidation_reason = "TAMPERED_REGION_ELIMINATED_BY_CROP"

        return TransformResult(
            image=cropped_img,
            mask=cropped_mask,
            metadata=self.get_metadata(),
            is_valid_comparison=is_valid,
            invalidation_reason=invalidation_reason,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="center_crop",
            severity_level=self.crop_ratio,
            severity_label=f"ratio={self.crop_ratio:.2f}",
            changes_geometry=True,
            changes_dimensions=True,
            requires_mask_interpolation=False,
            mask_interpolation_mode="nearest",
            parameters={"crop_ratio": self.crop_ratio},
            limitations=(
                "Changes image canvas dimensions. May remove peripheral tampered regions; "
                "invalidates comparison if ground truth is completely excised."
            ),
        )
