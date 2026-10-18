"""TruFor (Trustworthy Forgery Detection and Localization) model.

Reference:
    Guillaro et al. (CVPR 2023)
    "TruFor: leveraging all-round clues for trustworthy image forgery detection and localization"
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.models.checkpoint_hub import (
    extract_state_dict,
    get_online_checkpoint_info,
    resolve_or_download_checkpoint,
    validate_and_load_state_dict,
)
from transform_ai_detection.models.official_architectures import OfficialTruForNet
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
        q = self.q_proj(rgb_feat).view(b, c, h * w).permute(0, 2, 1)
        k = self.k_proj(noise_feat).view(b, c, h * w)
        v = self.v_proj(noise_feat).view(b, c, h * w).permute(0, 2, 1)

        scores = torch.matmul(q, k) / (c ** 0.5)
        attn = F.softmax(scores, dim=-1)
        fused = torch.matmul(attn, v).permute(0, 2, 1).contiguous().view(b, c, h, w)
        return rgb_feat + self.out_conv(fused)


class TruForNet(nn.Module):
    """Dual-branch RGB + Noiseprint++ cross-modal architecture for TruFor."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.noise_filter = ConstrainedNoiseFilter(3, 8, kernel_size=5)
        self.noise_stream = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = CrossModalAttention(dim=32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
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
    """Adapter for TruFor neural network architecture (Guillaro et al., CVPR 2023).

    Watches for pretrained checkpoints locally and online (Hugging Face DaniilOr/detect,
    official GRIP-UNINA archive, and IMDL-BenCo). Refuses to run with randomly initialized weights.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
        threshold: float = 0.5,
        require_checkpoint: bool = True,
        auto_download: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.requested_device = device
        self.device = get_torch_device(device if isinstance(device, str) else str(device))
        self.threshold = threshold
        self.require_checkpoint = True  # Random weight execution is strictly prohibited
        self.auto_download = auto_download
        self.cache_dir = cache_dir

        self._model: nn.Module = TruForNet().to(self.device)
        self._model.eval()

        self._is_loaded = False
        self._random_init_blocked = False
        self._load_error: Optional[str] = None

        self._try_load_model()

    @property
    def name(self) -> str:
        return "trufor"

    def is_available(self) -> bool:
        return self._is_loaded

    def _try_load_model(self) -> bool:
        resolved_path, err_notice = resolve_or_download_checkpoint(
            model_name=self.name,
            checkpoint_path=self.checkpoint_path,
            auto_download=self.auto_download,
            cache_dir=self.cache_dir,
        )
        if resolved_path is None:
            self._load_error = err_notice
            self._random_init_blocked = True
            self._is_loaded = False
            logger.warning(f"[{self.name}] {self._load_error}")
            return False

        try:
            logger.info(f"Loading TruFor checkpoint from {resolved_path}")
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            state_dict = extract_state_dict(ckpt)

            # Check if official TruFor (detconfcmx mit_b2 + DnCNN) keys are present
            if any(k.startswith(("backbone.patch_embed", "dncnn.", "decode_head.linear_c4")) for k in state_dict):
                official_net = OfficialTruForNet()
                ok, val_err = validate_and_load_state_dict(official_net, state_dict, self.name)
                if ok:
                    self._model = official_net.to(self.device)
                    self._model.eval()
                    self.checkpoint_path = resolved_path
                    self._is_loaded = True
                    self._random_init_blocked = False
                    self._load_error = None
                    return True

            ok, val_err = validate_and_load_state_dict(self._model, state_dict, self.name)
            if not ok:
                self._load_error = val_err
                self._random_init_blocked = True
                self._is_loaded = False
                logger.warning(f"[{self.name}] {self._load_error}")
                return False

            self._model.to(self.device)
            self._model.eval()
            self.checkpoint_path = resolved_path
            self._is_loaded = True
            self._random_init_blocked = False
            self._load_error = None
            return True
        except Exception as e:
            self._load_error = (
                f"Quitting with notice: Failed to load TruFor checkpoint from '{resolved_path}': {e}. "
                f"Running with randomly initialized weights is prohibited."
            )
            self._random_init_blocked = True
            self._is_loaded = False
            logger.warning(f"[{self.name}] {self._load_error}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        online_info = get_online_checkpoint_info(self.name)
        return {
            "name": self.name,
            "family": "deep_cross_modal_transformer",
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "random_init_blocked": self._random_init_blocked,
            "status_notice": self._load_error,
            "load_error": self._load_error,
            "online_checkpoint_available": online_info.get("online_available", False),
            "online_checkpoint_source": online_info.get("primary_source_desc"),
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
                f"TruFor model is not available. {self._load_error}"
            )

        start_t = time.perf_counter()

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
