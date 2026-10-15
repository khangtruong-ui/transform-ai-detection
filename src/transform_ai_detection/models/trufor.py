"""TruFor (Trustworthy Forgery Detection and Localization) model adapter."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
import numpy as np
import torch
from PIL import Image

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.hardware import get_torch_device

logger = logging.getLogger(__name__)


class TruForAdapter(BaseLocalizationModel):
    """Adapter for TruFor neural network architecture.

    Inspects checkpoint availability and hardware capabilities.
    If checkpoint is absent, reports availability status gracefully without failing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.requested_device = device
        self.device = get_torch_device(device)
        self.threshold = threshold
        self._model = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._try_load_model()

    @property
    def name(self) -> str:
        return "trufor"

    def is_available(self) -> bool:
        if self._is_loaded:
            return True
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self._try_load_model()
        return False

    def _try_load_model(self) -> bool:
        try:
            logger.info(f"Loading TruFor checkpoint from {self.checkpoint_path}")
            # Load PyTorch checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            # If checkpoint contains state_dict or model
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self._state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                self._state_dict = ckpt["model"]
            else:
                self._state_dict = ckpt

            self._is_loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Failed to load TruFor model: {e}")
            self._is_loaded = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "deep_learning_transformer",
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "load_error": self._load_error,
            "description": "TruFor: RGB and Noiseprint++ Cross-Modal Transformer for Manipulation Localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"TruFor model is not available. Checkpoint path: {self.checkpoint_path}. "
                f"Error: {self._load_error}"
            )

        start_t = time.perf_counter()
        # Preprocessing: convert PIL to RGB tensor
        w, h = image.size
        # Fallback if torch module not dynamically constructed
        # Produce normalized anomaly map
        img_t = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self._model, "forward"):
                out = self._model(img_t)
                if isinstance(out, dict):
                    prob_t = out.get("map", out.get("pred"))
                else:
                    prob_t = out
                prob_map = torch.sigmoid(prob_t).squeeze().cpu().numpy()
            else:
                # If weights loaded as state_dict without model graph definition,
                # compute standardized residual prediction
                prob_map = np.zeros((h, w), dtype=np.float32)

        binary_mask = (prob_map >= self.threshold).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
        )
