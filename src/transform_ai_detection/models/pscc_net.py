"""PSCC-Net: Progressive Spatio-Channel Correlation Network for IML.

Reference:
    Liu et al. (IEEE TCSVT 2022)
    "PSCC-Net: progressive spatio-channel correlation network for image manipulation
    detection and localization"
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
from transform_ai_detection.models.official_architectures import OfficialPSCCNet
from transform_ai_detection.hardware import get_torch_device

logger = logging.getLogger(__name__)


class SpatialCorrelationModule(nn.Module):
    """Computes dense spatial correlation across local patches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_theta = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_phi = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_g = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_out = nn.Conv2d(channels // 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        theta = self.conv_theta(x).view(b, c // 2, h * w).permute(0, 2, 1)
        phi = self.conv_phi(x).view(b, c // 2, h * w)
        g = self.conv_g(x).view(b, c // 2, h * w).permute(0, 2, 1)

        corr = torch.matmul(theta, phi) / ((c // 2) ** 0.5)
        attn = F.softmax(corr, dim=-1)

        out = torch.matmul(attn, g).permute(0, 2, 1).contiguous().view(b, c // 2, h, w)
        out = self.norm(self.conv_out(out))
        return x + out


class ChannelCorrelationModule(nn.Module):
    """Computes channel-wise correlation and recalibration for manipulation clues."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class PSCCNet(nn.Module):
    """Progressive Spatio-Channel Correlation Network."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.scm2 = SpatialCorrelationModule(32)
        self.ccm2 = ChannelCorrelationModule(32)
        self.scm3 = SpatialCorrelationModule(64)
        self.ccm3 = ChannelCorrelationModule(64)

        self.refine3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        c2 = self.ccm2(self.scm2(s2))
        c3 = self.ccm3(self.scm3(s3))

        r3 = self.refine3(c3)
        r3_up = F.interpolate(r3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        r2 = self.refine2(torch.cat([c2, r3_up], dim=1))
        r2_up = F.interpolate(r2, size=s1.shape[2:], mode="bilinear", align_corners=False)
        out = self.refine1(torch.cat([s1, r2_up], dim=1))

        out = out[:, :, :h, :w]
        return torch.sigmoid(out)


class PSCCNetModel(BaseLocalizationModel):
    """Adapter model for PSCC-Net (Liu et al., IEEE TCSVT 2022).

    Watches for pretrained checkpoints locally and online (GitHub proteus1991/PSCC-Net,
    Hugging Face Hub, and Google Drive). Refuses to run with randomly initialized weights.
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

        self.model: nn.Module = PSCCNet().to(self.device)
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
            logger.info(f"Loading PSCC-Net weights from {resolved_path}")
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)

            # Case 1: Official multi-component PSCC-Net checkpoint (HRNet + NLCDetection + DetectionHead)
            if isinstance(ckpt, dict) and "pscc_official_components" in ckpt:
                official_net = OfficialPSCCNet()
                official_net.load_official_components(ckpt["pscc_official_components"])
                self.model = official_net.to(self.device)
                self.model.eval()
                self.checkpoint_path = resolved_path
                self._is_loaded = True
                self._random_init_blocked = False
                self._load_error = None
                return True

            state_dict = extract_state_dict(ckpt)

            # Case 2: Official unified PSCC-Net state_dict
            if any(k.startswith(("FENet.", "SegNet.", "transition1.", "getmask4.")) for k in state_dict):
                official_net = OfficialPSCCNet()
                ok, val_err = validate_and_load_state_dict(official_net, state_dict, self.name)
                if ok:
                    self.model = official_net.to(self.device)
                    self.model.eval()
                    self.checkpoint_path = resolved_path
                    self._is_loaded = True
                    self._random_init_blocked = False
                    self._load_error = None
                    return True

            # Case 3: Compact PSCCNet state_dict
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
                f"Quitting with notice: Failed to load PSCC-Net checkpoint from '{resolved_path}': {e}. "
                f"Running with randomly initialized weights is prohibited."
            )
            self._random_init_blocked = True
            self._is_loaded = False
            logger.warning(f"[{self.name}] {self._load_error}")
            return False

    @property
    def name(self) -> str:
        return "pscc_net"

    def is_available(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        online_info = get_online_checkpoint_info(self.name)
        return {
            "name": self.name,
            "family": "progressive_spatio_channel_correlation_cnn",
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
                "correlation": "spatial + channel",
                "stages": 4,
            },
            "paper_reference": "Liu et al. (IEEE TCSVT 2022) PSCC-Net: progressive spatio-channel correlation network",
            "description": "PSCC-Net: Progressive spatio-channel correlation network for manipulation localization.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        if not self.is_available():
            raise RuntimeError(
                f"PSCC-Net model is not available. {self._load_error}"
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
            metadata={"device": str(self.device), "model_family": "PSCC-Net"},
        )
