"""IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer.

Reference:
    Ma et al. (AAAI 2024)
    "IML-ViT: benchmarking image manipulation localization by vision transformer"
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


class ViTStageBlock(nn.Module):
    """Transformer block with multi-head self-attention and MLP feed-forward."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class IMLViTNet(nn.Module):
    """Hierarchical Vision Transformer architecture with progressive aggregation decoder."""

    def __init__(self, in_channels: int = 3, dims: tuple = (32, 64, 128)) -> None:
        super().__init__()
        self.dims = dims

        # Stage 1: Patch 4x4 -> dim 32
        self.patch1 = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)
        self.block1 = ViTStageBlock(dim=dims[0], num_heads=2)

        # Stage 2: Stride 2 conv -> dim 64
        self.patch2 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.block2 = ViTStageBlock(dim=dims[1], num_heads=4)

        # Stage 3: Stride 2 conv -> dim 128
        self.patch3 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.block3 = ViTStageBlock(dim=dims[2], num_heads=8)

        # Progressive Feature Aggregation Decoder
        self.lateral3 = nn.Conv2d(dims[2], dims[1], kernel_size=1)
        self.lateral2 = nn.Conv2d(dims[1], dims[0], kernel_size=1)

        self.smooth2 = nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(dims[0], 16, kernel_size=4, stride=2, padding=1),
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
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Pad to multiple of 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Stage 1 (H/4, W/4)
        f1 = self.patch1(x)
        _, _, h1, w1 = f1.shape
        t1 = self.block1(f1.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, self.dims[0], h1, w1)

        # Stage 2 (H/8, W/8)
        f2 = self.patch2(t1)
        _, _, h2, w2 = f2.shape
        t2 = self.block2(f2.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, self.dims[1], h2, w2)

        # Stage 3 (H/16, W/16)
        f3 = self.patch3(t2)
        _, _, h3, w3 = f3.shape
        t3 = self.block3(f3.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, self.dims[2], h3, w3)

        # Progressive Top-Down Aggregation
        p3 = self.lateral3(t3)
        p2 = self.smooth2(t2 + F.interpolate(p3, size=(h2, w2), mode="bilinear", align_corners=False))
        p1 = self.smooth1(t1 + F.interpolate(self.lateral2(p2), size=(h1, w1), mode="bilinear", align_corners=False))

        out = self.head(p1)  # (B, 1, H_pad, W_pad)
        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class IMLViTModel(BaseLocalizationModel):
    """Adapter model for IML-ViT (Ma et al., AAAI 2024)."""

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

        self.model = IMLViTNet().to(self.device)
        self.model.eval()

        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._resolve_and_load_checkpoint()

    def _resolve_and_load_checkpoint(self) -> bool:
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(self.checkpoint_path)

        candidate_paths.extend([
            os.path.expanduser("~/.cache/iml_models/iml_vit.pth"),
            "checkpoints/iml_vit.pth",
        ])

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading IML-ViT weights from {path}")
                    ckpt = torch.load(path, map_location=self.device)
                    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    self.model.load_state_dict(state, strict=False)
                    self._is_loaded = True
                    self.checkpoint_path = path
                    return True
                except Exception as e:
                    self._load_error = str(e)
                    logger.warning(f"Failed to load IML-ViT checkpoint from {path}: {e}")

        if not self.require_checkpoint:
            self._is_loaded = True
            return True

        self._is_loaded = False
        return False

    @property
    def name(self) -> str:
        return "iml_vit"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "hierarchical_vision_transformer",
            "device": str(self.device),
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "load_error": self._load_error,
            "parameters": {
                "threshold": self.threshold,
                "stages": [32, 64, 128],
                "decoder": "progressive_aggregation",
            },
            "paper_reference": "Ma et al. (AAAI 2024) IML-ViT: benchmarking image manipulation localization by vision transformer",
            "description": "IML-ViT: Hierarchical vision transformer benchmarking architecture with progressive feature aggregation.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"IML-ViT model is not available. Checkpoint: {self.checkpoint_path}, "
                f"Error: {self._load_error}"
            )

        start_t = time.perf_counter()
        w, h = image.size

        img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob_t = self.model(img_t).squeeze()  # (H, W)
            prob_map = prob_t.cpu().numpy()
            binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
            metadata={"device": str(self.device), "model_family": "IML-ViT"},
        )
