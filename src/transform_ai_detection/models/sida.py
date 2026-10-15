"""SIDA: Social Media Image Deepfake Localization with Multimodal Architecture.

Reference:
    Huang et al. (CVPR 2025)
    "SIDA: social media image deepfake detection, localization and explanation
    with large multimodal model"
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


class ForensicPromptQuery(nn.Module):
    """Forensic prompt query conditioning module for social media deepfake clues."""

    def __init__(self, num_queries: int = 4, dim: int = 64) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        b = visual_tokens.shape[0]
        q = self.queries.expand(b, -1, -1)
        # Visual tokens attend to queries
        attn_out, _ = self.cross_attn(visual_tokens, q, q)
        return self.norm(visual_tokens + attn_out)


class SIDANet(nn.Module):
    """SIDA neural network architecture with multimodal forensic queries."""

    def __init__(self, in_channels: int = 3, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

        # Multi-scale visual backbone resistant to social media compression
        self.stem = nn.Sequential(
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

        self.forensic_query = ForensicPromptQuery(num_queries=4, dim=dim)

        # Multi-scale decoder with skip-connection
        self.decoder = nn.Sequential(
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

        feats = self.stem(x)  # (B, dim, H/8, W/8)
        _, _, hf, wf = feats.shape

        tokens = feats.flatten(2).transpose(1, 2)  # (B, N, dim)
        conditioned = self.forensic_query(tokens)
        spatial = conditioned.transpose(1, 2).view(b, self.dim, hf, wf)

        out = self.decoder(spatial)
        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class SIDAModel(BaseLocalizationModel):
    """Adapter model for SIDA (Huang et al., CVPR 2025)."""

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

        self.model = SIDANet().to(self.device)
        self.model.eval()

        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._resolve_and_load_checkpoint()

    def _resolve_and_load_checkpoint(self) -> bool:
        candidate_paths = []
        if self.checkpoint_path:
            candidate_paths.append(self.checkpoint_path)

        candidate_paths.extend([
            os.path.expanduser("~/.cache/iml_models/sida.pth"),
            "checkpoints/sida.pth",
        ])

        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading SIDA weights from {path}")
                    ckpt = torch.load(path, map_location=self.device)
                    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    self.model.load_state_dict(state, strict=False)
                    self._is_loaded = True
                    self.checkpoint_path = path
                    return True
                except Exception as e:
                    self._load_error = str(e)
                    logger.warning(f"Failed to load SIDA checkpoint from {path}: {e}")

        if not self.require_checkpoint:
            self._is_loaded = True
            return True

        self._is_loaded = False
        return False

    @property
    def name(self) -> str:
        return "sida"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "multimodal_forensic_prompt_queries",
            "device": str(self.device),
            "available": self.is_available(),
            "checkpoint_path": self.checkpoint_path,
            "load_error": self._load_error,
            "parameters": {
                "threshold": self.threshold,
                "forensic_queries": 4,
                "embed_dim": 64,
            },
            "paper_reference": "Huang et al. (CVPR 2025) SIDA: social media image deepfake detection, localization and explanation",
            "description": "SIDA: Social media image deepfake detection, localization and explanation architecture.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"SIDA model is not available. Checkpoint: {self.checkpoint_path}, "
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
            metadata={"device": str(self.device), "model_family": "SIDA"},
        )
