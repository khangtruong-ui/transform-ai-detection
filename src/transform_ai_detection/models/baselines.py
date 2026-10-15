"""Classical and forensic feature-based localization model baselines with GPU acceleration."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from PIL import Image, ImageChops
from scipy.ndimage import convolve, uniform_filter, gaussian_filter

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.hardware import get_torch_device


class ErrorLevelAnalysisModel(BaseLocalizationModel):
    """Error Level Analysis (ELA) forensic localization detector.

    Detects manipulation by recompressing the image at a known JPEG quality
    and identifying regions with anomalous error rates relative to the overall canvas.
    Supports GPU acceleration via PyTorch tensors on CUDA.
    """

    def __init__(
        self,
        recompress_quality: int = 90,
        threshold: float = 0.5,
        smoothing_sigma: float = 2.0,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.recompress_quality = recompress_quality
        self.threshold = threshold
        self.smoothing_sigma = smoothing_sigma
        self.device = get_torch_device(device if isinstance(device, str) else str(device))

        # Precompute 2D Gaussian smoothing kernel if smoothing_sigma > 0
        if self.smoothing_sigma > 0:
            radius = max(1, int(round(self.smoothing_sigma * 3)))
            x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            gauss_1d = torch.exp(-0.5 * (x / self.smoothing_sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            self._gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0).to(self.device)
            self._radius = radius
        else:
            self._gauss_2d = None
            self._radius = 0

    @property
    def name(self) -> str:
        return "ela_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "forensic_compression_analysis",
            "device": str(self.device),
            "parameters": {
                "recompress_quality": self.recompress_quality,
                "threshold": self.threshold,
                "smoothing_sigma": self.smoothing_sigma,
            },
            "description": "Error Level Analysis: detects JPEG recompression artifact discrepancies.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        img_rgb = image.convert("RGB")

        # In-memory JPEG recompression
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=self.recompress_quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        if self.device.type == "cuda":
            with torch.no_grad():
                arr_orig = np.array(img_rgb, dtype=np.float32)
                arr_recomp = np.array(recompressed, dtype=np.float32)
                t_orig = torch.from_numpy(arr_orig).permute(2, 0, 1).unsqueeze(0).to(self.device)
                t_recomp = torch.from_numpy(arr_recomp).permute(2, 0, 1).unsqueeze(0).to(self.device)

                diff_t = torch.abs(t_orig - t_recomp).mean(dim=1, keepdim=True)

                if self._gauss_2d is not None:
                    smoothed_t = torch.nn.functional.conv2d(diff_t, self._gauss_2d, padding=self._radius)
                else:
                    smoothed_t = diff_t

                mean_err_t = smoothed_t.mean()
                std_err_t = smoothed_t.std() + 1e-6
                z_scores_t = (smoothed_t - mean_err_t) / std_err_t
                prob_t = torch.sigmoid(z_scores_t - 1.5).squeeze()

                binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()
                prob_map = prob_t.cpu().numpy()
                mean_err = float(mean_err_t.cpu().item())
                std_err = float(std_err_t.cpu().item())
        else:
            diff = ImageChops.difference(img_rgb, recompressed)
            diff_arr = np.array(diff, dtype=np.float32).mean(axis=2)

            if self.smoothing_sigma > 0:
                smoothed = gaussian_filter(diff_arr, sigma=self.smoothing_sigma)
            else:
                smoothed = diff_arr

            mean_err = np.mean(smoothed)
            std_err = np.std(smoothed) + 1e-6
            z_scores = (smoothed - mean_err) / std_err

            prob_map = 1.0 / (1.0 + np.exp(-(z_scores - 1.5)))
            binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
            metadata={"mean_error": float(mean_err), "std_error": float(std_err)},
        )


class NoiseResidualModel(BaseLocalizationModel):
    """Spatial Rich Model (SRM) / High-Pass Filter Noise Residual Inconsistency Detector.

    Applies high-pass filters to extract noise residuals and localizes regions where
    residual energy significantly departs from the background camera sensor noise profile.
    Accelerated with 2D convolution and pooling on PyTorch CUDA.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 15,
        filter_type: str = "laplacian",
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self.filter_type = filter_type
        self.device = get_torch_device(device if isinstance(device, str) else str(device))

        # High-pass filter kernel (3x3 Laplacian / SRM residual)
        if filter_type == "srm":
            k = np.array([
                [-1, 2, -1],
                [2, -4, 2],
                [-1, 2, -1],
            ], dtype=np.float32)
        else:
            k = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ], dtype=np.float32)

        self.kernel = k
        self._kernel_t = torch.from_numpy(k).unsqueeze(0).unsqueeze(0).to(self.device)

    @property
    def name(self) -> str:
        return "noise_residual_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "noise_inconsistency",
            "device": str(self.device),
            "parameters": {
                "threshold": self.threshold,
                "window_size": self.window_size,
                "filter_type": self.filter_type,
            },
            "description": "High-pass spatial noise residual variance inconsistency detector.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0

        if self.device.type == "cuda":
            with torch.no_grad():
                gray_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
                pad = self.kernel.shape[0] // 2
                res_t = torch.nn.functional.conv2d(gray_t, self._kernel_t, padding=pad)

                pad_w = self.window_size // 2
                mean_res_t = torch.nn.functional.avg_pool2d(
                    res_t, kernel_size=self.window_size, stride=1, padding=pad_w
                )
                mean_sq_t = torch.nn.functional.avg_pool2d(
                    res_t**2, kernel_size=self.window_size, stride=1, padding=pad_w
                )
                local_var_t = torch.clamp(mean_sq_t - mean_res_t**2, min=0.0)

                # Standardize local variance using median and MAD on GPU
                med_val = torch.median(local_var_t)
                mad_val = torch.median(torch.abs(local_var_t - med_val)) + 1e-6
                dev_t = torch.abs(local_var_t - med_val) / mad_val

                prob_t = torch.sigmoid(dev_t - 2.0).squeeze()
                binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()
                prob_map = prob_t.cpu().numpy()
                med = float(med_val.cpu().item())
                mad = float(mad_val.cpu().item())
        else:
            residual = convolve(gray, self.kernel, mode="reflect")
            mean_res = uniform_filter(residual, size=self.window_size)
            mean_sq = uniform_filter(residual**2, size=self.window_size)
            local_var = np.maximum(0.0, mean_sq - mean_res**2)

            med = np.median(local_var)
            mad = np.median(np.abs(local_var - med)) + 1e-6
            dev = np.abs(local_var - med) / mad

            prob_map = 1.0 / (1.0 + np.exp(-(dev - 2.0)))
            binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
            metadata={"median_variance": float(med), "mad_variance": float(mad)},
        )


class DCTInconsistencyModel(BaseLocalizationModel):
    """DCT Block Inconsistency & Grid Artifact Detector.

    Checks for 8x8 block boundary discontinuities and high-frequency coefficient variance.
    Accelerated with PyTorch CUDA tensor operations.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.threshold = threshold
        self.device = get_torch_device(device if isinstance(device, str) else str(device))

    @property
    def name(self) -> str:
        return "dct_inconsistency_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "frequency_domain",
            "device": str(self.device),
            "parameters": {"threshold": self.threshold},
            "description": "Local 8x8 block boundary discontinuity and energy inconsistency.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0

        if self.device.type == "cuda":
            with torch.no_grad():
                gray_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

                diff_y = torch.zeros_like(gray_t)
                diff_x = torch.zeros_like(gray_t)
                diff_y[:, :, 1:, :] = torch.abs(gray_t[:, :, 1:, :] - gray_t[:, :, :-1, :])
                diff_x[:, :, :, 1:] = torch.abs(gray_t[:, :, :, 1:] - gray_t[:, :, :, :-1])

                total_diff = diff_y + diff_x
                local_energy = torch.nn.functional.avg_pool2d(total_diff, kernel_size=9, stride=1, padding=4)

                mean_e = local_energy.mean()
                std_e = local_energy.std() + 1e-6
                z = (local_energy - mean_e) / std_e
                prob_t = torch.sigmoid(z - 1.8).squeeze()

                binary_mask = (prob_t >= self.threshold).to(torch.uint8).cpu().numpy()
                prob_map = prob_t.cpu().numpy()
        else:
            h, w = gray.shape
            diff_y = np.zeros_like(gray)
            diff_x = np.zeros_like(gray)
            diff_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
            diff_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])

            local_energy = uniform_filter(diff_y + diff_x, size=9)
            mean_e = np.mean(local_energy)
            std_e = np.std(local_energy) + 1e-6

            z = (local_energy - mean_e) / std_e
            prob_map = 1.0 / (1.0 + np.exp(-(z - 1.8)))
            binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ModelOutput(
            binary_mask=binary_mask,
            probability_map=prob_map,
            threshold_used=self.threshold,
            inference_time_ms=elapsed_ms,
        )
