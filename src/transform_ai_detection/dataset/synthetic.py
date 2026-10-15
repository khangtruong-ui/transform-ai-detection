"""Synthetic dataset adapter for offline testing, CI/CD, and fast experiments."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw

from transform_ai_detection.dataset.base import BaseDatasetAdapter, DatasetSample
from transform_ai_detection.dataset.validator import DatasetValidator


class SyntheticDatasetAdapter(BaseDatasetAdapter):
    """Generates deterministic synthetic image manipulation samples offline."""

    def __init__(
        self,
        num_samples: int = 20,
        image_size: tuple[int, int] = (256, 256),
        seed: int = 42,
        validator: Optional[DatasetValidator] = None,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.validator = validator or DatasetValidator()

        self._methods = ["authentic", "synthetic_splice", "synthetic_inpaint", "synthetic_copymove"]
        self._sample_ids = [f"synth_{i:04d}" for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def get_sample_id(self, index: int) -> str:
        return self._sample_ids[index]

    def __getitem__(self, index: int) -> DatasetSample:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} out of range [0, {self.num_samples})")

        sample_id = self._sample_ids[index]
        rng = np.random.RandomState(self.seed + index * 1001)

        w, h = self.image_size
        method = self._methods[index % len(self._methods)]
        is_manipulated = method != "authentic"
        label = 1 if is_manipulated else 0
        label_text = "fake" if is_manipulated else "real"

        # Generate base synthetic image (multi-frequency background)
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        base = 128 + 60 * np.sin(xx + index) * np.cos(yy + index * 0.5)
        noise = rng.normal(0, 10, (h, w, 3))
        img_arr = np.clip(np.repeat(base[:, :, None], 3, axis=2) + noise, 0, 255).astype(np.uint8)

        mask_arr = np.zeros((h, w), dtype=np.uint8)

        if is_manipulated:
            # Create a tampered region (ellipse or polygon)
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)

            cx = rng.randint(w // 4, 3 * w // 4)
            cy = rng.randint(h // 4, 3 * h // 4)
            rx = rng.randint(w // 8, w // 4)
            ry = rng.randint(h // 8, h // 4)

            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
            mask_arr = (np.array(mask_img) > 127).astype(np.uint8)

            # Manipulate image pixels within mask
            if method == "synthetic_splice":
                # Splice high-frequency texture inside mask
                splice_noise = rng.uniform(0, 255, (h, w, 3)).astype(np.uint8)
                img_arr[mask_arr == 1] = splice_noise[mask_arr == 1]
            elif method == "synthetic_inpaint":
                # Smooth / blur inside mask
                mean_color = img_arr[mask_arr == 1].mean(axis=0) if mask_arr.sum() > 0 else [128, 128, 128]
                img_arr[mask_arr == 1] = mean_color.astype(np.uint8)
            elif method == "synthetic_copymove":
                # Copy from elsewhere
                shift_x = min(rx, w // 4)
                shifted = np.roll(img_arr, shift=shift_x, axis=1)
                img_arr[mask_arr == 1] = shifted[mask_arr == 1]

        pil_img = Image.fromarray(img_arr, mode="RGB")
        mask_ratio = float(mask_arr.sum() / mask_arr.size)

        # Validate
        exclusion_reason = self.validator.validate_raw(
            sample_id=sample_id,
            index=index,
            image_obj=pil_img,
            mask_arr=mask_arr,
            label=label,
            label_text=label_text,
            mask_ratio_reported=mask_ratio,
        )

        metadata: Dict[str, Any] = {
            "source_index": index,
            "exclusion_reason": exclusion_reason,
            "is_valid": exclusion_reason is None,
        }

        return DatasetSample(
            sample_id=sample_id,
            image=pil_img,
            mask=mask_arr,
            label=label,
            label_text=label_text,
            dataset_source="SyntheticBench",
            method=method,
            mask_ratio_reported=mask_ratio,
            metadata=metadata,
        )
