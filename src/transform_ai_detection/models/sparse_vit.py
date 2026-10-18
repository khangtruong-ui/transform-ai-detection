"""SparseViT: Non-semantics-Centered Parameter-Efficient IML model.

Reference:
    Su et al. (AAAI 2025)
    "Can we get rid of handcrafted feature extractors? SparseViT: nonsemantics-centered,
    parameter-efficient image manipulation localization through spare-coding transformer"
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
from transform_ai_detection.models.official_architectures import OfficialSparseViTNet
from transform_ai_detection.hardware import get_torch_device

logger = logging.getLogger(__name__)


class SparseAttentionBlock(nn.Module):
    """Sparse-coding multi-head attention block."""

    def __init__(self, dim: int = 64, num_heads: int = 4, sparsity_ratio: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.num_heads
        d = self.head_dim

        norm_x = self.norm1(x)
        q = self.q_proj(norm_x).view(b, n, h, d).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, n, h, d).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, n, h, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

        k_keep = max(1, int(n * (1.0 - self.sparsity_ratio)))
        topk_vals, _ = torch.topk(scores, k=k_keep, dim=-1)
        threshold = topk_vals[..., -1, None]
        sparse_mask = scores < threshold
        scores = scores.masked_fill(sparse_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, c)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class SparseViTNet(nn.Module):
    """PyTorch architecture for SparseViT Image Manipulation Localizer."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = 8
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=8)
        self.blocks = nn.ModuleList([
            SparseAttentionBlock(dim=embed_dim, num_heads=4, sparsity_ratio=0.5)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 32, kernel_size=4, stride=2, padding=1),
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
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.patch_embed(x)
        _, _, hp, wp = feats.shape

        tokens = feats.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        spatial_feats = tokens.transpose(1, 2).view(b, self.embed_dim, hp, wp)
        out = self.decoder(spatial_feats)

        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class SparseViTModel(BaseLocalizationModel):
    """Adapter model for SparseViT (Su et al., AAAI 2025).

    Watches for pretrained checkpoints locally and online (scu-zjz/SparseViT Google Drive).
    Refuses to run with randomly initialized weights.
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

        self.model: nn.Module = SparseViTNet().to(self.device)
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
            logger.info(f"Loading SparseViT weights from {resolved_path}")
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            state_dict = extract_state_dict(ckpt)

            # Check if official SparseViT_Mul keys are present
            if any(k.startswith(("encoder_net.patch_embed", "lmu.gamma")) for k in state_dict):
                official_net = OfficialSparseViTNet()
                ok, val_err = validate_and_load_state_dict(official_net, state_dict, self.name)
                if ok:
                    self.model = official_net.to(self.device)
                    self.model.eval()
                    self.checkpoint_path = resolved_path
                    self._is_loaded = True
                    self._random_init_blocked = False
                    self._load_error = None
                    return True

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
                f"Quitting with notice: Failed to load SparseViT checkpoint from '{resolved_path}': {e}. "
                f"Running with randomly initialized weights is prohibited."
            )
            self._random_init_blocked = True
            self._is_loaded = False
            logger.warning(f"[{self.name}] {self._load_error}")
            return False

    @property
    def name(self) -> str:
        return "sparsevit"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        online_info = get_online_checkpoint_info(self.name)
        return {
            "name": self.name,
            "family": "sparse_coding_vision_transformer",
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
                "embed_dim": 64,
                "patch_size": 8,
                "sparsity_ratio": 0.5,
            },
            "paper_reference": "Su et al. (AAAI 2025) Can we get rid of handcrafted feature extractors? SparseViT",
            "description": "SparseViT: Non-semantics-centered sparse coding transformer for manipulation localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"SparseViT model is not available. {self._load_error}"
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
            metadata={"device": str(self.device), "model_family": "SparseViT"},
        )
