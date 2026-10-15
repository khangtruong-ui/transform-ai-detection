"""Unit tests for models and model registry."""

from __future__ import annotations

import numpy as np
import pytest

from transform_ai_detection.models.baselines import (
    ErrorLevelAnalysisModel,
    NoiseResidualModel,
    DCTInconsistencyModel,
)
from transform_ai_detection.models.synthetic import SyntheticOracleModel, MockConstantModel
from transform_ai_detection.models.registry import ModelRegistry


def test_error_level_analysis_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = ErrorLevelAnalysisModel()
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() and out.probability_map.max() <= 1.0
    assert set(np.unique(out.binary_mask)).issubset({0, 1})


def test_noise_residual_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = NoiseResidualModel()
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map is not None


def test_dct_inconsistency_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = DCTInconsistencyModel()
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)


def test_model_registry_and_availability():
    models = ["ela_detector", "noise_residual_detector", "mock_zeros"]
    report = ModelRegistry.generate_availability_report(models)
    assert len(report) == 3
    for entry in report:
        assert entry["registered"] is True
        assert entry["load_successful"] is True
