"""Base localization model interfaces and standardized model output."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


@dataclass
class ModelOutput:
    """Standardized output produced by a localization model."""

    binary_mask: np.ndarray  # (H, W) uint8 {0, 1}
    probability_map: Optional[np.ndarray] = None  # (H, W) float32 [0.0, 1.0] or None
    threshold_used: float = 0.5
    inference_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.binary_mask.ndim == 3:
            self.binary_mask = self.binary_mask[:, :, 0]
        if self.binary_mask.max() > 1:
            self.binary_mask = (self.binary_mask > 127).astype(np.uint8)
        else:
            self.binary_mask = (self.binary_mask > 0).astype(np.uint8)

        if self.probability_map is not None:
            if self.probability_map.ndim == 3:
                self.probability_map = self.probability_map[:, :, 0]
            self.probability_map = np.clip(self.probability_map, 0.0, 1.0).astype(np.float32)


class BaseLocalizationModel(abc.ABC):
    """Abstract interface for image manipulation localization (IML) models."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, image: Image.Image) -> ModelOutput:
        """Run inference on a single PIL image."""
        raise NotImplementedError

    def predict_batch(self, images: List[Image.Image]) -> List[ModelOutput]:
        """Run batch inference. Default implementation performs sequential prediction."""
        return [self.predict(img) for img in images]

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if model weights and dependencies are available."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata, architecture description, and hardware requirements."""
        raise NotImplementedError
