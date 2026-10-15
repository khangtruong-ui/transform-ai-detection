"""Dataset adapters and validation module."""

from transform_ai_detection.dataset.base import BaseDatasetAdapter, DatasetSample
from transform_ai_detection.dataset.validator import DatasetValidator, ExclusionLogger, ExclusionRecord
from transform_ai_detection.dataset.synthetic import SyntheticDatasetAdapter
from transform_ai_detection.dataset.tampered_combined import TamperedCombinedDatasetAdapter

__all__ = [
    "BaseDatasetAdapter",
    "DatasetSample",
    "DatasetValidator",
    "ExclusionLogger",
    "ExclusionRecord",
    "SyntheticDatasetAdapter",
    "TamperedCombinedDatasetAdapter",
]
