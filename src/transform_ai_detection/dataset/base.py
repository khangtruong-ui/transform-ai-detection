"""Base dataset adapter interfaces and unified sample representation."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
from PIL import Image


@dataclass
class DatasetSample:
    """Unified representation of an image manipulation localization sample."""

    sample_id: str
    image: Image.Image
    mask: np.ndarray  # 2D binary uint8 array (H, W), values {0, 1}
    label: int  # 0: authentic / real, 1: manipulated / fake
    label_text: str  # "real" or "fake"
    dataset_source: str
    method: str
    mask_ratio_reported: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mask, np.ndarray):
            self.mask = np.asarray(self.mask, dtype=np.uint8)
        if self.mask.ndim == 3:
            self.mask = self.mask[:, :, 0]
        # Normalize to strictly {0, 1}
        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.uint8)
        else:
            self.mask = (self.mask > 0).astype(np.uint8)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def is_manipulated(self) -> bool:
        return bool(self.label == 1 or self.mask.sum() > 0)

    @property
    def computed_mask_ratio(self) -> float:
        total = self.mask.size
        return float(self.mask.sum() / total) if total > 0 else 0.0


class BaseDatasetAdapter(abc.ABC):
    """Abstract interface for dataset adapters."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Total number of available samples."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> DatasetSample:
        """Fetch a single sample by its integer index."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[DatasetSample]:
        for idx in range(len(self)):
            yield self[idx]

    @abc.abstractmethod
    def get_sample_id(self, index: int) -> str:
        """Get sample ID without necessarily decoding full payloads."""
        raise NotImplementedError

    def deterministic_sample_indices(
        self,
        count: Optional[int] = None,
        seed: int = 42,
        indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Produce a deterministic, reproducible list of sample indices."""
        total = len(self)
        if indices is not None:
            # Validate indices
            valid = [i for i in indices if 0 <= i < total]
            return valid

        all_indices = list(range(total))
        if count is None or count >= total:
            return all_indices

        rng = random.Random(seed)
        return rng.sample(all_indices, count)
