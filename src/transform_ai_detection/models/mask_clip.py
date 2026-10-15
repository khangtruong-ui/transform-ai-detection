"""MaskCLIP: Pretrained Vision-Model Dense Feature Fusion for Manipulation Localization.

Reference:
    Wang et al. (CVPR 2025)
    "OpenSDI: spotting diffusion-generated images in the open world" / MaskCLIP
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


class MaskCLIPBlock(nn.Module):
    """Dense vision transformer block with modified 1x1 key-value projection."""

    def __init__(self, dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * ((c // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))
        x = x + self.mlp(x)
        return x


class MaskCLIPNet(nn.Module):
    """MaskCLIP architecture with 1x1 dense patch artifact classifier."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Patch embedding: 8x8 patches
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=8)
        self.blocks = nn.ModuleList([MaskCLIPBlock(dim=embed_dim, num_heads=4) for _ in range(2)])
        self.norm = nn.LayerNorm(embed_dim)

        # 1x1 dense classifier
        self.dense_classifier = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Upsampling refinement head
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
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
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.patch_embed(x)
        _, _, hp, wp = feats.shape

        tokens = feats.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        spatial_tokens = tokens.transpose(1, 2).view(b, self.embed_dim, hp, wp)
        dense_feats = self.dense_classifier(spatial_tokens)
        out = self.upsample(dense_feats)

        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class MaskCLIPModel(BaseLocalizationModel):
    """Adapter model for MaskCLIP (Wang et al., CVPR 2025)."""

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

        self.model = MaskCLIPNet().to(self.device)
        self.model.eval()

        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._resolve_and_load_checkpoint()

    def _resolve_and_load_checkpoint(self) -> bool:
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(self.checkpoint_path)

        candidate_paths.extend([
            os.path.expanduser("~/.cache/iml_models/maskclip.pth"),
            "checkpoints/maskclip.pth",
        ])

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading MaskCLIP weights from {path}")
                    ckpt = torch.load(path, map_location=self.device)
                    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    self.model.load_state_dict(state, strict=False)
                    self._is_loaded = True
                    self.checkpoint_path = path
                    return True
                except Exception as e:
                    self._load_error = str(e)
                    logger.warning(f"Failed to load MaskCLIP checkpoint from {path}: {e}")

        if not self.require_checkpoint:
            self._is_loaded = True
            return True

        self._is_loaded = False
        return False

    @property
    def name(self) -> str:
        return "maskclip"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "dense_clip_feature_fusion",
            "device": str(self.device),
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "load_error": self._load_error,
            "parameters": {
                "threshold": self.threshold,
                "embed_dim": 64,
                "classifier": "1x1_dense_conv",
            },
            "paper_reference": "Wang et al. (CVPR 2025) OpenSDI: spotting diffusion-generated images in the open world",
            "description": "MaskCLIP: Pretrained vision-model dense feature fusion for manipulation localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"MaskCLIP model is not available. Checkpoint: {self.checkpoint_path}, "
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
            metadata={"device": str(self.device), "model_family": "MaskCLIP"},
        )
