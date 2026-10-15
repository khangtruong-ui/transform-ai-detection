"""Adapter for KhangTruong/Tampered-Images-Combined-Dataset using parquet-dataset-loader."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
try:
    import parquet_dataset_loader as pdl
except ImportError:
    pdl = None

from transform_ai_detection.dataset.base import BaseDatasetAdapter, DatasetSample
from transform_ai_detection.dataset.validator import DatasetValidator

logger = logging.getLogger(__name__)


class TamperedCombinedDatasetAdapter(BaseDatasetAdapter):
    """Dataset adapter for KhangTruong/Tampered-Images-Combined-Dataset.

    Uses `parquet-dataset-loader` for instant metadata indexing, bounded memory,
    and HTTP Range-based streaming without downloading large files to disk.
    """

    can_instant_sample_ids: bool = False

    def __init__(
        self,
        repo_id: str = "KhangTruong/Tampered-Images-Combined-Dataset",
        split: str = "cross_test",
        streaming: bool = True,
        max_cached_row_groups: int = 2,
        validator: Optional[DatasetValidator] = None,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.streaming = streaming
        self.max_cached_row_groups = max_cached_row_groups
        self.validator = validator or DatasetValidator()
        self._sample_id_cache: Dict[int, str] = {}
        self._id_dataset: Optional[Any] = None

        # Silence excessive urllib3 connection pool warnings during shard resolution
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

        if pdl is None:
            raise ImportError(
                "The 'parquet-dataset-loader' package is required for TamperedCombinedDatasetAdapter. "
                "Install it or use SyntheticDatasetAdapter instead."
            )
        logger.info(f"Initializing Parquet dataset from {repo_id} (split: {split})")
        self._dataset = pdl.load_dataset(
            self.repo_id,
            split=self.split,
            streaming=self.streaming,
            max_cached_row_groups=self.max_cached_row_groups,
        )
        self._length = len(self._dataset)
        logger.info(f"Loaded {self.repo_id} with {self._length:,} rows.")

    def __len__(self) -> int:
        return self._length

    def get_sample_id(self, index: int) -> str:
        if index in self._sample_id_cache:
            return self._sample_id_cache[index]

        # Use projected reader to transfer only 'id' column rather than heavy image/mask payloads
        if self._id_dataset is None:
            self._id_dataset = self._dataset.select_columns(["id"])
        row = self._id_dataset[index]
        sid = str(row.get("id", f"sample_{index}"))
        self._sample_id_cache[index] = sid
        return sid

    def __getitem__(self, index: int) -> DatasetSample:
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range [0, {self._length})")

        row = self._dataset[index]
        sample_id = str(row.get("id", f"sample_{index}"))
        self._sample_id_cache[index] = sample_id

        # Decode image
        image_data = row.get("image")
        if isinstance(image_data, dict) and "bytes" in image_data:
            img = Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        elif isinstance(image_data, Image.Image):
            img = image_data.convert("RGB")
        else:
            raise ValueError(f"Unrecognized image data format for sample {sample_id}")

        # Decode mask
        mask_data = row.get("mask")
        if isinstance(mask_data, dict) and "bytes" in mask_data:
            mask_pil = Image.open(io.BytesIO(mask_data["bytes"]))
            mask_arr = np.array(mask_pil)
        elif isinstance(mask_data, Image.Image):
            mask_arr = np.array(mask_data)
        elif isinstance(mask_data, np.ndarray):
            mask_arr = mask_data
        else:
            raise ValueError(f"Unrecognized mask data format for sample {sample_id}")

        # Handle mask channels: if RGB or 3D, take 1st channel or mean
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

        # Binarize mask to uint8 {0, 1}
        # In this dataset, fake regions are marked > 127 (or > 0 for binary masks)
        if mask_arr.max() > 1:
            bin_mask = (mask_arr > 127).astype(np.uint8)
        else:
            bin_mask = (mask_arr > 0).astype(np.uint8)

        label = int(row.get("label", 1 if bin_mask.sum() > 0 else 0))
        label_text = str(row.get("label_text", "fake" if label == 1 else "real"))
        dataset_source = str(row.get("dataset", "unknown"))
        method = str(row.get("method", "unknown"))
        mask_ratio_reported = float(row.get("mask_ratio", 0.0))

        # Validate
        exclusion_reason = self.validator.validate_raw(
            sample_id=sample_id,
            index=index,
            image_obj=img,
            mask_arr=bin_mask,
            label=label,
            label_text=label_text,
            mask_ratio_reported=mask_ratio_reported,
        )

        metadata: Dict[str, Any] = {
            "source_index": index,
            "exclusion_reason": exclusion_reason,
            "is_valid": exclusion_reason is None,
        }

        return DatasetSample(
            sample_id=sample_id,
            image=img,
            mask=bin_mask,
            label=label,
            label_text=label_text,
            dataset_source=dataset_source,
            method=method,
            mask_ratio_reported=mask_ratio_reported,
            metadata=metadata,
        )
