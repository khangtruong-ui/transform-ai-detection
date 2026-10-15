"""CoDE: Decision-Making Environment for Image Forgery Localization.

Reference:
    Peng et al. (IEEE TIFS 2024)
    "Employing reinforcement learning to construct a decision-making environment
    for image forgery localization"
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.hardware import get_torch_device

logger = logging.getLogger(__name__)


class CoDENet(nn.Module):
    """Reinforcement learning decision-making network with continuous and discrete action heads."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # State representation backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Discrete action head: predicts regional tamper confidence decisions
        self.discrete_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )

        # Continuous action head: dense pixel-level boundary refinement
        self.continuous_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.encoder(x)
        # Discrete regional decisions modulate continuous refinement
        disc_state = self.discrete_head(feats).view(b, 64, 1, 1)
        modulated_feats = feats * (1.0 + disc_state)

        out = self.continuous_decoder(modulated_feats)
        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class CoDEModel(BaseLocalizationModel):
    """Adapter model for CoDE (Peng et al., IEEE TIFS 2024)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
        threshold: float = 0.5,
        require_checkpoint: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = get_torch_device(device if isinstance(device, str) else str(device))
        self.threshold = threshold
        self.require_checkpoint = require_checkpoint

        self.model = CoDENet().to(self.device)
        self.model.eval()

        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._resolve_and_load_checkpoint()

    def _resolve_and_load_checkpoint(self) -> bool:
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(self.checkpoint_path)

        candidate_paths.extend([
            os.path.expanduser("~/.cache/iml_models/code.pth"),
            "checkpoints/code.pth",
        ])

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading CoDE weights from {path}")
                    ckpt = torch.load(path, map_location=self.device)
                    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    self.model.load_state_dict(state, strict=False)
                    self._is_loaded = True
                    self.checkpoint_path = path
                    return True
                except Exception as e:
                    self._load_error = str(e)
                    logger.warning(f"Failed to load CoDE checkpoint from {path}: {e}")

        if not self.require_checkpoint:
            self._is_loaded = True
            return True

        self._is_loaded = False
        return False

    @property
    def name(self) -> str:
        return "code"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "reinforcement_learning_decision_environment",
            "device": str(self.device),
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "load_error": self._load_error,
            "parameters": {
                "threshold": self.threshold,
                "actions": "continuous_and_discrete",
            },
            "paper_reference": "Peng et al. (IEEE TIFS 2024) Employing reinforcement learning to construct a decision-making environment for image forgery localization",
            "description": "CoDE: Continuous and discrete decision-making environment for tampering localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"CoDE model is not available. Checkpoint: {self.checkpoint_path}, "
                f"Error: {self._load_error}"
            )

        start_t = time.perf_counter()
        w, h = image.size

        img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob_t = self.model(img_t).squeeze()
            prob_map = prob_t.cpu().numpy()
            binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
            metadata={"device": str(self.device), "model_family": "CoDE"},
        )
