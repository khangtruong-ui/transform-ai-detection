"""Denoising, sharpening, and additive noise transformations."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from transform_ai_detection.transformations.base import (
    BaseTransformation,
    TransformMetadata,
    TransformResult,
    TransformationRegistry,
)


@TransformationRegistry.register("gaussian_noise")
class GaussianNoiseTransformation(BaseTransformation):
    """Additive zero-mean Gaussian noise on RGB pixel channels."""

    def __init__(self, std: float = 0.03) -> None:
        self.std = float(std)

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        # Deterministic RNG derived from seed and sample_id
        import hashlib
        h = int(hashlib.md5(f"{seed}_{sample_id}_{self.std}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)

        arr = np.array(image, dtype=np.float32) / 255.0
        noise = rng.normal(0, self.std, arr.shape)
        noisy_arr = np.clip(arr + noise, 0.0, 1.0)
        noisy_img = Image.fromarray((noisy_arr * 255.0).astype(np.uint8), mode="RGB")

        return TransformResult(
            image=noisy_img,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="gaussian_noise",
            severity_level=self.std,
            severity_label=f"std={self.std:.3f}",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={"std": self.std},
            limitations="Corrupts high-frequency camera sensor noise signatures.",
        )


@TransformationRegistry.register("sharpening")
class SharpeningTransformation(BaseTransformation):
    """Image sharpening via unsharp masking or enhancement factor."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = float(factor)

    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        sharpened = ImageEnhance.Sharpness(image).enhance(self.factor)
        return TransformResult(
            image=sharpened,
            mask=np.copy(mask),
            metadata=self.get_metadata(),
            is_valid_comparison=True,
        )

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name="sharpening",
            severity_level=self.factor,
            severity_label=f"sharp={self.factor:.1f}",
            changes_geometry=False,
            changes_dimensions=False,
            requires_mask_interpolation=False,
            mask_interpolation_mode="none",
            parameters={"factor": self.factor},
            limitations="Amplifies high-frequency edge gradients and residual noise.",
        )
