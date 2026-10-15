"""Unit tests for dataset adapters, validation, and exclusion tracking."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transform_ai_detection.dataset.synthetic import SyntheticDatasetAdapter
from transform_ai_detection.dataset.validator import DatasetValidator, ExclusionLogger


def test_synthetic_dataset_adapter():
    adapter = SyntheticDatasetAdapter(num_samples=10, image_size=(64, 64), seed=42)
    assert len(adapter) == 10

    sample = adapter[0]
    assert sample.width == 64
    assert sample.height == 64
    assert sample.mask.shape == (64, 64)
    assert set(np.unique(sample.mask)).issubset({0, 1})
    assert sample.label in (0, 1)
    assert sample.label_text in ("real", "fake")
    assert sample.dataset_source == "SyntheticBench"


def test_deterministic_sample_selection():
    adapter = SyntheticDatasetAdapter(num_samples=50, seed=123)
    s1 = adapter.deterministic_sample_indices(count=10, seed=999)
    s2 = adapter.deterministic_sample_indices(count=10, seed=999)
    s3 = adapter.deterministic_sample_indices(count=10, seed=888)

    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 10


def test_validator_detects_dimension_mismatch():
    validator = DatasetValidator()
    img = Image.new("RGB", (100, 100))
    mask = np.zeros((80, 80), dtype=np.uint8)

    reason = validator.validate_raw(
        sample_id="test_mismatch",
        index=0,
        image_obj=img,
        mask_arr=mask,
        label=0,
        label_text="real",
        mask_ratio_reported=0.0,
    )
    assert reason == "DIMENSION_MISMATCH"
    assert "DIMENSION_MISMATCH" in validator.exclusion_logger.summary()


def test_validator_detects_duplicate_id():
    validator = DatasetValidator()
    img = Image.new("RGB", (100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)

    r1 = validator.validate_raw("dup_1", 0, img, mask, 0, "real", 0.0)
    assert r1 is None

    r2 = validator.validate_raw("dup_1", 1, img, mask, 0, "real", 0.0)
    assert r2 == "DUPLICATE_SAMPLE_ID"


def test_validator_detects_authentic_with_nonempty_mask():
    validator = DatasetValidator()
    img = Image.new("RGB", (100, 100))
    # Nonempty mask for authentic image
    mask = np.ones((100, 100), dtype=np.uint8)

    reason = validator.validate_raw("bad_real", 0, img, mask, 0, "real", 0.0)
    assert reason == "AUTHENTIC_IMAGE_NONEMPTY_MASK"


def test_dataset_adapter_instant_flags():
    from transform_ai_detection.dataset.synthetic import SyntheticDatasetAdapter
    from transform_ai_detection.dataset.tampered_combined import TamperedCombinedDatasetAdapter

    assert SyntheticDatasetAdapter.can_instant_sample_ids is True
    assert TamperedCombinedDatasetAdapter.can_instant_sample_ids is False


def test_deterministic_sample_selection_is_sorted():
    adapter = SyntheticDatasetAdapter(num_samples=50, seed=123)
    s = adapter.deterministic_sample_indices(count=10, seed=42)
    assert s == sorted(s)
