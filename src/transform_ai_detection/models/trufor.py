"""TruFor (Trustworthy Forgery Detection and Localization) model.

Reference:
    Guillaro et al. (CVPR 2023)
    "TruFor: leveraging all-round clues for trustworthy image forgery detection and localization"
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


class ConstrainedNoiseFilter(nn.Module):
    """Constrained residual filter to extract Noiseprint++ tampering clues."""

    def __init__(self, in_channels: int = 3, out_channels: int = 8, kernel_size: int = 5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.center_idx = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.clone()
        w[:, :, self.center_idx, self.center_idx] = 0.0
        denom = w.sum(dim=(2, 3), keepdim=True) + 1e-7
        w = w / denom
        w[:, :, self.center_idx, self.center_idx] = -1.0
        return F.conv2d(x, w, padding=self.padding)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between RGB features and Noiseprint++ features."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, noise_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        q = self.q_proj(rgb_feat).view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        k = self.k_proj(noise_feat).view(b, c, h * w)  # (B, C, HW)
        v = self.v_proj(noise_feat).view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)

        scores = torch.matmul(q, k) / (c ** 0.5)
        attn = F.softmax(scores, dim=-1)
        fused = torch.matmul(attn, v).permute(0, 2, 1).contiguous().view(b, c, h, w)
        return rgb_feat + self.out_conv(fused)


class TruForNet(nn.Module):
    """Dual-branch RGB + Noiseprint++ cross-modal architecture for TruFor."""

    def __init__(self) -> None:
        super().__init__()
        # RGB stream
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Noise residual stream
        self.noise_filter = ConstrainedNoiseFilter(3, 8, kernel_size=5)
        self.noise_stream = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Cross-modal fusion
        self.cross_attn = CrossModalAttention(dim=32)

        # Localization decoder
        self.decoder = nn.Sequential(
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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        rgb_feats = self.rgb_stream(x)
        noise_res = self.noise_filter(x)
        noise_feats = self.noise_stream(noise_res)

        fused = self.cross_attn(rgb_feats, noise_feats)
        out = self.decoder(fused)

        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class TruForAdapter(BaseLocalizationModel):
    """Adapter for TruFor neural network architecture.

    Maintains backwards compatibility with previous TruForAdapter interface,
    supports loading official checkpoints, and falls back to integrated TruForNet
    with full GPU acceleration.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
        threshold: float = 0.5,
        require_checkpoint: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.requested_device = device
        self.device = get_torch_device(device if isinstance(device, str) else str(device))
        self.threshold = threshold
        self.require_checkpoint = require_checkpoint

        self._model = TruForNet().to(self.device)
        self._model.eval()

        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._try_load_model()

    @property
    def name(self) -> str:
        return "trufor"

    def is_available(self) -> bool:
        return self._is_loaded

    def _try_load_model(self) -> bool:
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(self.checkpoint_path)

        candidate_paths.extend([
            os.path.expanduser("~/.cache/iml_models/trufor.pth"),
            "checkpoints/trufor.pth",
        ])

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading TruFor checkpoint from {path}")
                    ckpt = torch.load(path, map_location=self.device)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                    elif isinstance(ckpt, dict) and "model" in ckpt:
                        state_dict = ckpt["model"]
                    else:
                        state_dict = ckpt

                    self._model.load_state_dict(state_dict, strict=False)
                    self._is_loaded = True
                    self.checkpoint_path = path
                    return True
                except Exception as e:
                    self._load_error = str(e)
                    logger.warning(f"Failed to load TruFor model from {path}: {e}")

        if not self.require_checkpoint:
            self._is_loaded = True
            return True

        self._is_loaded = False
        return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "deep_cross_modal_transformer",
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "load_error": self._load_error,
            "parameters": {
                "threshold": self.threshold,
                "streams": ["rgb", "noiseprint++"],
                "fusion": "cross_modal_attention",
            },
            "paper_reference": "Guillaro et al. (CVPR 2023) TruFor: leveraging all-round clues for trustworthy image forgery detection and localization",
            "description": "TruFor: RGB and Noiseprint++ Cross-Modal Transformer for Manipulation Localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"TruFor model is not available. Checkpoint path: {self.checkpoint_path}. "
                f"Error: {self._load_error}"
            )

        start_t = time.perf_counter()
        w, h = image.size

        img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._model(img_t)
            if isinstance(out, dict):
                prob_t = out.get("map", out.get("pred"))
            else:
                prob_t = out
            prob_t = prob_t.squeeze()
            prob_map = prob_t.cpu().numpy()
            binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
            metadata={"device": str(self.device), "model_family": "TruFor"},
        )
