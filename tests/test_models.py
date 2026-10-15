"""Unit tests for models and model registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from transform_ai_detection.models.baselines import (
    ErrorLevelAnalysisModel,
    NoiseResidualModel,
    DCTInconsistencyModel,
)
from transform_ai_detection.models.deep_forensic import DeepForensicModel, BayarConv2d
from transform_ai_detection.models.synthetic import SyntheticOracleModel, MockConstantModel
from transform_ai_detection.models.registry import ModelRegistry


def test_error_level_analysis_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = ErrorLevelAnalysisModel(device="cpu")
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() and out.probability_map.max() <= 1.0
    assert set(np.unique(out.binary_mask)).issubset({0, 1})


def test_noise_residual_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = NoiseResidualModel(device="cpu")
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map is not None


def test_dct_inconsistency_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = DCTInconsistencyModel(device="cpu")
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)


def test_deep_forensic_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepForensicModel(device=device)
    assert model.is_available()

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() and out.probability_map.max() <= 1.0


def test_bayar_conv2d_constraint():
    layer = BayarConv2d(in_channels=3, out_channels=4, kernel_size=5)
    x = torch.randn(2, 3, 32, 32)
    out = layer(x)
    assert out.shape == (2, 4, 32, 32)


def test_gpu_execution_when_available(synthetic_image_and_mask):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    img, _ = synthetic_image_and_mask
    for ModelCls in [NoiseResidualModel, ErrorLevelAnalysisModel, DCTInconsistencyModel, DeepForensicModel]:
        mod = ModelCls(device="cuda")
        out = mod.predict(img)
        assert out.binary_mask.shape == (img.height, img.width)
        assert str(mod.device).startswith("cuda")


def test_model_registry_and_availability():
    models = ["ela_detector", "noise_residual_detector", "deep_forensic_detector", "mock_zeros"]
    report = ModelRegistry.generate_availability_report(models, default_device="auto")
    assert len(report) == 4
    for entry in report:
        assert entry["registered"] is True
        assert entry["load_successful"] is True
