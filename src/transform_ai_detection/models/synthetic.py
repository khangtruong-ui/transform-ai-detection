"""Deterministic synthetic and mock models for testing, CI/CD, and ablation studies."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput


class SyntheticOracleModel(BaseLocalizationModel):
    """Synthetic model that produces deterministic pseudo-predictions.

    Can simulate high-accuracy, moderate-accuracy, or degrading behavior without
    requiring deep neural network weights.
    """

    def __init__(
        self,
        name: str = "synthetic_oracle",
        noise_level: float = 0.05,
        threshold: float = 0.5,
    ) -> None:
        self._name = name
        self.noise_level = noise_level
        self.threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "synthetic_mock",
            "device": "cpu",
            "parameters": {"noise_level": self.noise_level, "threshold": self.threshold},
            "description": "Deterministic mock model for ablation and unit tests.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        w, h = image.size

        # Create deterministic pseudo-mask based on pixel content hash
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        # Use simple spatial derivative as pseudo-indicator
        diff = np.abs(gray - np.roll(gray, shift=1, axis=1))
        # Normalize
        norm_diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)

        # Seed from image content
        h_val = int(hashlib.md5(gray.tobytes()[:1000]).hexdigest()[:8], 16)
        rng = np.random.RandomState(h_val)
        noise = rng.normal(0, self.noise_level, (h, w))
        prob_map = np.clip(norm_diff + noise, 0.0, 1.0).astype(np.float32)

        binary_mask = (prob_map >= self.threshold).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
        )


class MockConstantModel(BaseLocalizationModel):
    """Mock model returning constant predictions (e.g. all 0s or all 1s)."""

    def __init__(self, constant_value: int = 0, name: str = "mock_constant") -> None:
        self.constant_value = constant_value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.name, "constant_value": self.constant_value}

    def predict(self, image: Image.Image) -> ModelOutput:
        w, h = image.size
        mask = np.full((h, w), self.constant_value, dtype=np.uint8)
        prob = np.full((h, w), float(self.constant_value), dtype=np.float32)
        return ModelOutput(binary_mask=mask, probability_map=prob)
