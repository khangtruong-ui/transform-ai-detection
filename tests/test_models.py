"""Unit tests for models and model registry."""

from __future__ import annotations

import os
import tempfile
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
from transform_ai_detection.models.sparse_vit import SparseViTModel, SparseViTNet
from transform_ai_detection.models.iml_vit import IMLViTModel, IMLViTNet
from transform_ai_detection.models.pscc_net import PSCCNetModel, PSCCNet
from transform_ai_detection.models.trufor import TruForAdapter, TruForNet
from transform_ai_detection.models.code_net import CoDEModel, CoDENet
from transform_ai_detection.models.mask_clip import MaskCLIPModel, MaskCLIPNet
from transform_ai_detection.models.sida import SIDAModel, SIDANet
from transform_ai_detection.models.rita import RITAModel, RITANet


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


# --- Tests for 8 Integrated Models ---

def test_sparse_vit_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = SparseViTModel(device="cpu")
    assert model.is_available()
    assert model.name == "sparsevit"

    info = model.get_model_info()
    assert "Su et al. (AAAI 2025)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0
    assert set(np.unique(out.binary_mask)).issubset({0, 1})


def test_iml_vit_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = IMLViTModel(device="cpu")
    assert model.is_available()
    assert model.name == "iml_vit"

    info = model.get_model_info()
    assert "Ma et al. (AAAI 2024)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_pscc_net_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = PSCCNetModel(device="cpu")
    assert model.is_available()
    assert model.name == "pscc_net"

    info = model.get_model_info()
    assert "Liu et al. (IEEE TCSVT 2022)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_trufor_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = TruForAdapter(device="cpu")
    assert model.is_available()
    assert model.name == "trufor"

    info = model.get_model_info()
    assert "Guillaro et al. (CVPR 2023)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_code_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = CoDEModel(device="cpu")
    assert model.is_available()
    assert model.name == "code"

    info = model.get_model_info()
    assert "Peng et al. (IEEE TIFS 2024)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_mask_clip_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = MaskCLIPModel(device="cpu")
    assert model.is_available()
    assert model.name == "maskclip"

    info = model.get_model_info()
    assert "Wang et al. (CVPR 2025)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_sida_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = SIDAModel(device="cpu")
    assert model.is_available()
    assert model.name == "sida"

    info = model.get_model_info()
    assert "Huang et al. (CVPR 2025)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_rita_model(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    model = RITAModel(device="cpu")
    assert model.is_available()
    assert model.name == "rita"

    info = model.get_model_info()
    assert "Zhu et al. (CVPR Findings 2026)" in info["paper_reference"]

    out = model.predict(img)
    assert out.binary_mask.shape == (img.height, img.width)
    assert out.probability_map.shape == (img.height, img.width)
    assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0


def test_all_paper_models_gpu(synthetic_image_and_mask):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    img, _ = synthetic_image_and_mask
    paper_model_classes = [
        SparseViTModel,
        IMLViTModel,
        PSCCNetModel,
        TruForAdapter,
        CoDEModel,
        MaskCLIPModel,
        SIDAModel,
        RITAModel,
    ]

    for ModelCls in paper_model_classes:
        mod = ModelCls(device="cuda")
        assert mod.is_available()
        assert str(mod.device).startswith("cuda")
        out = mod.predict(img)
        assert out.binary_mask.shape == (img.height, img.width)
        assert out.inference_time_ms > 0


def test_model_registry_creates_all_8_models():
    expected_models = [
        "sparsevit",
        "iml_vit",
        "pscc_net",
        "trufor",
        "code",
        "maskclip",
        "sida",
        "rita",
    ]
    report = ModelRegistry.generate_availability_report(expected_models, default_device="cpu")
    assert len(report) == 8
    for entry in report:
        assert entry["registered"] is True
        assert entry["load_successful"] is True
        assert entry["available"] is True


def test_checkpoint_saving_and_loading(synthetic_image_and_mask):
    img, _ = synthetic_image_and_mask
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_sparsevit.pth")
        net = SparseViTNet()
        torch.save({"state_dict": net.state_dict()}, ckpt_path)

        # Load with checkpoint_path
        model = SparseViTModel(checkpoint_path=ckpt_path, device="cpu", require_checkpoint=True)
        assert model.is_available()
        out = model.predict(img)
        assert out.binary_mask.shape == (img.height, img.width)


def test_require_checkpoint_flag():
    # If checkpoint doesn't exist and require_checkpoint=True, model should report unavailable
    model = SparseViTModel(checkpoint_path="/nonexistent/path.pth", require_checkpoint=True)
    assert not model.is_available()
    with pytest.raises(RuntimeError):
        model.predict(None)
