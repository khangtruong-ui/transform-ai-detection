"""Dataset validation and exclusion tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExclusionRecord:
    sample_id: str
    index: int
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExclusionLogger:
    """Tracks and records excluded samples and validation anomalies."""

    def __init__(self) -> None:
        self.exclusions: List[ExclusionRecord] = []
        self.seen_ids: Set[str] = set()

    def record_exclusion(
        self, sample_id: str, index: int, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        rec = ExclusionRecord(
            sample_id=sample_id,
            index=index,
            reason=reason,
            details=details or {},
        )
        self.exclusions.append(rec)
        logger.warning(f"Exclusion [id={sample_id}, idx={index}]: {reason} ({details})")

    def is_duplicate(self, sample_id: str) -> bool:
        if sample_id in self.seen_ids:
            return True
        self.seen_ids.add(sample_id)
        return False

    def export_json(self, filepath: str) -> None:
        data = [e.to_dict() for e in self.exclusions]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.exclusions:
            counts[e.reason] = counts.get(e.reason, 0) + 1
        return counts


class DatasetValidator:
    """Validates individual samples and collects exclusion metadata."""

    def __init__(
        self,
        min_dimension: int = 16,
        ratio_tolerance: float = 0.05,
        allow_empty_fake_mask: bool = False,
    ) -> None:
        self.min_dimension = min_dimension
        self.ratio_tolerance = ratio_tolerance
        self.allow_empty_fake_mask = allow_empty_fake_mask
        self.exclusion_logger = ExclusionLogger()

    def validate_raw(
        self,
        sample_id: str,
        index: int,
        image_obj: Optional[Image.Image],
        mask_arr: Optional[np.ndarray],
        label: Optional[int],
        label_text: Optional[str],
        mask_ratio_reported: Optional[float],
    ) -> Optional[str]:
        """Validate raw components. Returns exclusion reason if invalid, or None if valid."""
        # 1. Duplicate ID check
        if self.exclusion_logger.is_duplicate(sample_id):
            self.exclusion_logger.record_exclusion(
                sample_id=sample_id,
                index=index,
                reason="DUPLICATE_SAMPLE_ID",
                details={"sample_id": sample_id},
            )
            return "DUPLICATE_SAMPLE_ID"

        # 2. Null payload check
        if image_obj is None or mask_arr is None:
            self.exclusion_logger.record_exclusion(
                sample_id=sample_id,
                index=index,
                reason="NULL_PAYLOAD",
                details={"image_is_none": image_obj is None, "mask_is_none": mask_arr is None},
            )
            return "NULL_PAYLOAD"

        # 3. Minimum dimensions
        w, h = image_obj.size
        if w < self.min_dimension or h < self.min_dimension:
            self.exclusion_logger.record_exclusion(
                sample_id=sample_id,
                index=index,
                reason="IMAGE_TOO_SMALL",
                details={"width": w, "height": h, "min_required": self.min_dimension},
            )
            return "IMAGE_TOO_SMALL"

        # 4. Dimension match between image and mask
        mask_h, mask_w = mask_arr.shape[:2]
        if (w, h) != (mask_w, mask_h):
            self.exclusion_logger.record_exclusion(
                sample_id=sample_id,
                index=index,
                reason="DIMENSION_MISMATCH",
                details={
                    "image_size": [w, h],
                    "mask_size": [mask_w, mask_h],
                },
            )
            return "DIMENSION_MISMATCH"

        # 5. Label and mask semantic check
        computed_ratio = float((mask_arr > 0).sum() / mask_arr.size)
        if label == 0:  # authentic / real
            # Authentic images should not have substantial tampered mask area
            if computed_ratio > 0.005:
                self.exclusion_logger.record_exclusion(
                    sample_id=sample_id,
                    index=index,
                    reason="AUTHENTIC_IMAGE_NONEMPTY_MASK",
                    details={"label": label, "computed_mask_ratio": computed_ratio},
                )
                return "AUTHENTIC_IMAGE_NONEMPTY_MASK"
        elif label == 1:  # fake / tampered
            if computed_ratio == 0.0 and not self.allow_empty_fake_mask:
                self.exclusion_logger.record_exclusion(
                    sample_id=sample_id,
                    index=index,
                    reason="FAKE_IMAGE_EMPTY_MASK",
                    details={"label": label, "computed_mask_ratio": computed_ratio},
                )
                return "FAKE_IMAGE_EMPTY_MASK"

        # Check mask ratio discrepancy (log warning if significant)
        if mask_ratio_reported is not None and mask_ratio_reported > 0:
            diff = abs(computed_ratio - mask_ratio_reported)
            # Some datasets report mask_ratio with different formulas (e.g. 1/N or bounding box),
            # so we note this in warning details rather than excluding unless configured
            if diff > self.ratio_tolerance:
                logger.debug(
                    f"Sample {sample_id}: reported ratio {mask_ratio_reported:.4f} vs "
                    f"computed ratio {computed_ratio:.4f} (diff: {diff:.4f})"
                )

        return None
