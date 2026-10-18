"""RITA: Process-Aware Autoregressive Localizer for Realistic Manipulation Scenarios.

Reference:
    Zhu et al. (CVPR Findings 2026)
    "Revisiting image manipulation localization under realistic manipulation scenarios"
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
from transform_ai_detection.hardware import get_torch_device

logger = logging.getLogger(__name__)


class ProcessAwareAutoregressiveBlock(nn.Module):
    """Causal autoregressive feature dependency modeling block."""

    def __init__(self, dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)

        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RITANet(nn.Module):
    """RITA neural network with process-aware autoregressive token modeling."""

    def __init__(self, in_channels: int = 3, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.ar_block1 = ProcessAwareAutoregressiveBlock(dim=dim, num_heads=4)
        self.ar_block2 = ProcessAwareAutoregressiveBlock(dim=dim, num_heads=4)

        self.boundary_refine = nn.Sequential(
            nn.ConvTranspose2d(dim, 32, kernel_size=4, stride=2, padding=1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.patch_proj(x)
        _, _, hf, wf = feats.shape

        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.ar_block1(tokens)
        tokens = self.ar_block2(tokens)

        spatial = tokens.transpose(1, 2).view(b, self.dim, hf, wf)
        out = self.boundary_refine(spatial)
        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class RITAModel(BaseLocalizationModel):
    """Adapter model for RITA (Zhu et al., CVPR Findings 2026).

    Note: Official weights (scu-zjz/RITA) are only released on password-protected
    Baidu Netdisk (https://pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn).
    If no local checkpoint is found, RITAModel quits with a clear notice and refuses
    to execute with randomly initialized weights.
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
        self.device = get_torch_device(device if isinstance(device, str) else str(device))
        self.threshold = threshold
        self.require_checkpoint = True  # Random weight execution is strictly prohibited
        self.auto_download = auto_download
        self.cache_dir = cache_dir

        self.model = RITANet().to(self.device)
        self.model.eval()

        self._is_loaded = False
        self._random_init_blocked = False
        self._load_error: Optional[str] = None

        self._resolve_and_load_checkpoint()

    def _resolve_and_load_checkpoint(self) -> bool:
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
            logger.info(f"Loading RITA weights from {resolved_path}")
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            state_dict = extract_state_dict(ckpt)
            ok, val_err = validate_and_load_state_dict(self.model, state_dict, self.name)
            if not ok:
                self._load_error = val_err
                self._random_init_blocked = True
                self._is_loaded = False
                logger.warning(f"[{self.name}] {self._load_error}")
                return False

            self.model.to(self.device)
            self.model.eval()
            self.checkpoint_path = resolved_path
            self._is_loaded = True
            self._random_init_blocked = False
            self._load_error = None
            return True
        except Exception as e:
            self._load_error = (
                f"Quitting with notice: Failed to load RITA checkpoint from '{resolved_path}': {e}. "
                f"Running with randomly initialized weights is prohibited."
            )
            self._random_init_blocked = True
            self._is_loaded = False
            logger.warning(f"[{self.name}] {self._load_error}")
            return False

    @property
    def name(self) -> str:
        return "rita"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        online_info = get_online_checkpoint_info(self.name)
        return {
            "name": self.name,
            "family": "process_aware_autoregressive_transformer",
            "device": str(self.device),
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "random_init_blocked": self._random_init_blocked,
            "status_notice": self._load_error,
            "load_error": self._load_error,
            "online_checkpoint_available": online_info.get("online_available", False),
            "online_checkpoint_source": online_info.get("primary_source_desc"),
            "parameters": {
                "threshold": self.threshold,
                "autoregressive_blocks": 2,
                "embed_dim": 64,
            },
            "paper_reference": "Zhu et al. (CVPR Findings 2026) Revisiting image manipulation localization under realistic manipulation scenarios",
            "description": "RITA: Process-aware autoregressive localizer for realistic manipulation scenarios.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"RITA model is not available. {self._load_error}"
            )

        start_t = time.perf_counter()

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
            metadata={"device": str(self.device), "model_family": "RITA"},
        )
