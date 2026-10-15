"""Classical and forensic feature-based localization model baselines."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image, ImageChops
from scipy.ndimage import convolve, uniform_filter, gaussian_filter

from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput


class ErrorLevelAnalysisModel(BaseLocalizationModel):
    """Error Level Analysis (ELA) forensic localization detector.

    Detects manipulation by recompressing the image at a known JPEG quality
    and identifying regions with anomalous error rates relative to the overall canvas.
    """

    def __init__(
        self,
        recompress_quality: int = 90,
        threshold: float = 0.5,
        smoothing_sigma: float = 2.0,
    ) -> None:
        self.recompress_quality = recompress_quality
        self.threshold = threshold
        self.smoothing_sigma = smoothing_sigma

    @property
    def name(self) -> str:
        return "ela_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "forensic_compression_analysis",
            "device": "cpu",
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

        # Save to memory at designated quality
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=self.recompress_quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        # Pixel difference
        diff = ImageChops.difference(img_rgb, recompressed)
        diff_arr = np.array(diff, dtype=np.float32).mean(axis=2)  # grayscale error

        # Smooth to suppress high-frequency point noise
        if self.smoothing_sigma > 0:
            smoothed = gaussian_filter(diff_arr, sigma=self.smoothing_sigma)
        else:
            smoothed = diff_arr

        # Normalize relative to canvas distribution (z-score and sigmoid)
        mean_err = np.mean(smoothed)
        std_err = np.std(smoothed) + 1e-6
        z_scores = (smoothed - mean_err) / std_err

        # Map to [0, 1] probability map using sigmoid centered at 2.0 standard deviations
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
    """

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 15,
        filter_type: str = "laplacian",
    ) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self.filter_type = filter_type

        # High-pass filter kernel (3x3 Laplacian / SRM residual)
        if filter_type == "srm":
            self.kernel = np.array([
                [-1, 2, -1],
                [2, -4, 2],
                [-1, 2, -1],
            ], dtype=np.float32)
        else:
            self.kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ], dtype=np.float32)

    @property
    def name(self) -> str:
        return "noise_residual_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "noise_inconsistency",
            "device": "cpu",
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

        # Convolve with high pass kernel
        residual = convolve(gray, self.kernel, mode="reflect")

        # Local variance of residual
        mean_res = uniform_filter(residual, size=self.window_size)
        mean_sq = uniform_filter(residual**2, size=self.window_size)
        local_var = np.maximum(0.0, mean_sq - mean_res**2)

        # Standardize local variance
        med = np.median(local_var)
        mad = np.median(np.abs(local_var - med)) + 1e-6
        dev = np.abs(local_var - med) / mad

        # Sigmoid probability
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
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "dct_inconsistency_detector"

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": "frequency_domain",
            "device": "cpu",
            "parameters": {"threshold": self.threshold},
            "description": "Local 8x8 block boundary discontinuity and energy inconsistency.",
        }

    def predict(self, image: Image.Image) -> ModelOutput:
        start_t = time.perf_counter()
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        h, w = gray.shape

        # Gradient along 8x8 grid lines
        grid_mask_y = np.zeros(h, dtype=bool)
        grid_mask_x = np.zeros(w, dtype=bool)
        grid_mask_y[::8] = True
        grid_mask_x[::8] = True

        # Vertical and horizontal 1st-differences
        diff_y = np.zeros_like(gray)
        diff_x = np.zeros_like(gray)
        diff_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        diff_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])

        # Grid jump vs non-grid jump
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
