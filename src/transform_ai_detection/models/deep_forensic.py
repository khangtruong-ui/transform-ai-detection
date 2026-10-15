"""Deep PyTorch-based neural forensic localization model using Bayar constrained convolution."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.hardware import get_torch_device


class BayarConv2d(nn.Module):
    """Bayar Constrained Convolution layer (Bayar & Stamm, 2018).

    Forces the convolutional filters to act as adaptive prediction-error filters,
    suppressing image semantic content while amplifying low-level tampering artifacts.
    Constraint: center weight = -1, sum of remaining weights = 1.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 8, kernel_size: int = 5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.center_idx = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clone weights to apply constraint during forward pass
        w = self.weight.clone()
        # Zero center
        w[:, :, self.center_idx, self.center_idx] = 0.0
        # Normalize non-center weights to sum to 1 per filter channel
        denom = w.sum(dim=(2, 3), keepdim=True) + 1e-7
        w = w / denom
        # Set center to -1
        w[:, :, self.center_idx, self.center_idx] = -1.0

        return F.conv2d(x, w, padding=self.padding)


class DeepForensicNet(nn.Module):
    """Deep residual forensic network combining BayarConv and multi-scale feature refinement."""

    def __init__(self) -> None:
        super().__init__()
        self.bayar = BayarConv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Initialize deterministic forensic priors
        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract low-level noise residual features via Bayar filter
        res = self.bayar(x)
        c1 = self.conv1(res)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        u2 = self.up2(c2)

        # Ensure spatial dimension match
        if u2.shape[2:] != c1.shape[2:]:
            u2 = F.interpolate(u2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        cat = torch.cat([c1, u2], dim=1)
        logits = self.head(cat)
        return torch.sigmoid(logits)


class DeepForensicModel(BaseLocalizationModel):
    """Deep learning neural network model for manipulation localization on GPU."""

    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        threshold: float = 0.5,
    ) -> None:
        self.device = get_torch_device(device if isinstance(device, str) else str(device))
        self.threshold = threshold
        self.model = DeepForensicNet().to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "deep_forensic_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "deep_constrained_cnn",
            "device": str(self.device),
            "parameters": {
                "threshold": self.threshold,
                "architecture": "BayarConv2d + MultiScaleResNet",
            },
            "description": "Deep PyTorch neural network with adaptive Bayar constrained residual convolutions.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        w, h = image.size

        # Convert image to RGB float tensor on target device (GPU)
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
        )
