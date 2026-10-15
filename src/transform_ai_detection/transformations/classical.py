"""Classical image transformations with explicit severity and geometry declarations."""

from __future__ import annotations

import io
from typing import Any, Dict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from transform_ai_detection.transformations.base import (
    BaseTransformation,
    TransformMetadata,
    TransformResult,
    TransformationRegistry,
)


@TransformationRegistry.register("identity")
class IdentityTransformation(BaseTransformation):
    """Identity baseline transformation (no changes)."""

    def __init__(self, severity_level: int = 0) -> None:
        self.severity_level = severity_level

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        # Return exact copy
        return TransformResult(
            image=image.copy(),
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="identity",
            severity_level=self.severity_level,
            severity_label="original",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={},
            limitations="Reference baseline. No transformation applied.",
        )


@TransformationRegistry.register("jpeg_compression")
class JPEGCompressionTransformation(BaseTransformation):
    """Recompression via standard JPEG codec at specified quality level."""

    def __init__(self, quality: int = 75) -> None:
        self.quality = int(quality)
        if not (1 <= self.quality <= 100):
            raise ValueError(f"JPEG quality must be in range [1, 100], got {self.quality}")

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        buf = io.BytesIO()
        # Compress in memory
        image.save(buf, format="JPEG", quality=self.quality, optimize=True)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        return TransformResult(
            image=recompressed,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="jpeg_compression",
            severity_level=self.quality,
            severity_label=f"Q={self.quality}",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={"quality": self.quality},
            limitations=(
                "Introduces 8x8 DCT block artifacts and chroma subsampling. "
                "May suppress high-frequency forensic noise patterns."
            ),
        )


@TransformationRegistry.register("gaussian_blur")
class GaussianBlurTransformation(BaseTransformation):
    """Gaussian filtering with specified standard deviation (sigma/radius)."""

    def __init__(self, sigma: float = 2.0) -> None:
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise ValueError(f"Sigma must be non-negative, got {self.sigma}")

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        if self.sigma == 0:
            blurred = image.copy()
        else:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        return TransformResult(
            image=blurred,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="gaussian_blur",
            severity_level=self.sigma,
            severity_label=f"sigma={self.sigma:.1f}",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={"sigma": self.sigma},
            limitations=(
                "Low-pass filtering attenuates high-frequency sensor noise "
                "and synthetic generator boundary artifacts."
            ),
        )


@TransformationRegistry.register("downscale_upscale")
class DownscaleUpscaleTransformation(BaseTransformation):
    """Downscaling followed by resizing back to the original dimensions."""

    def __init__(self, scale_factor: float = 0.5) -> None:
        self.scale_factor = float(scale_factor)
        if not (0.01 <= self.scale_factor <= 1.0):
            raise ValueError(f"scale_factor must be in (0, 1], got {self.scale_factor}")

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        orig_w, orig_h = image.size
        low_w = max(1, int(round(orig_w * self.scale_factor)))
        low_h = max(1, int(round(orig_h * self.scale_factor)))

        # Downscale and restore with bicubic
        downscaled = image.resize((low_w, low_h), resample=Image.Resampling.BICUBIC)
        restored = downscaled.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)

        return TransformResult(
            image=restored,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="downscale_upscale",
            severity_level=self.scale_factor,
            severity_label=f"scale={self.scale_factor:.2f}",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={"scale_factor": self.scale_factor},
            limitations=(
                "Simulates resolution downsampling across web pipelines. "
                "Preserves original canvas geometry but destroys sub-pixel boundary traces."
            ),
        )


@TransformationRegistry.register("brightness_contrast")
class BrightnessContrastTransformation(BaseTransformation):
    """Adjusts luminance and dynamic range contrast."""

    def __init__(
        self,
        factor_brightness: float = 1.0,
        factor_contrast: float = 1.0,
    ) -> None:
        self.factor_brightness = float(factor_brightness)
        self.factor_contrast = float(factor_contrast)

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        enhanced = image
        if self.factor_brightness != 1.0:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(self.factor_brightness)
        if self.factor_contrast != 1.0:
            enhanced = ImageEnhance.Contrast(enhanced).enhance(self.factor_contrast)

        return TransformResult(
            image=enhanced,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        severity_label = f"B={self.factor_brightness:.1f}_C={self.factor_contrast:.1f}"
        return TransformMetadata(
            name="brightness_contrast",
            severity_level={"brightness": self.factor_brightness, "contrast": self.factor_contrast},
            severity_label=severity_label,
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={
                "factor_brightness": self.factor_brightness,
                "factor_contrast": self.factor_contrast,
            },
            limitations=(
                "May clip highlight or shadow detail, potentially altering "
                "luminance consistency across manipulated boundaries."
            ),
        )
