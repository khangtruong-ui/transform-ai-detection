"""Unit tests for models, online checkpoint hub, official architectures, and random-init blocking."""

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
from transform_ai_detection.models.deep_forensic import (
    DeepForensicModel,
    DeepForensicNet,
    BayarConv2d,
)
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
from transform_ai_detection.models.checkpoint_hub import (
    ONLINE_CHECKPOINT_REGISTRY,
    get_online_checkpoint_info,
    resolve_or_download_checkpoint,
    validate_and_load_state_dict,
)
from transform_ai_detection.models.official_architectures import (
    OfficialPSCCNet,
    OfficialCoDENet,
)


def _save_valid_checkpoint(net_cls, path: str) -> str:
    net = net_cls()
    torch.save({"state_dict": net.state_dict()}, path)
    return path


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


def test_bayar_conv2d_constraint():
    layer = BayarConv2d(in_channels=3, out_channels=4, kernel_size=5)
    x = torch.randn(2, 3, 32, 32)
    out = layer(x)
    assert out.shape == (2, 4, 32, 32)


@pytest.mark.parametrize(
    "model_cls,net_cls,expected_name,expected_ref",
    [
        (DeepForensicModel, DeepForensicNet, "deep_forensic_detector", None),
        (SparseViTModel, SparseViTNet, "sparsevit", "Su et al. (AAAI 2025)"),
        (IMLViTModel, IMLViTNet, "iml_vit", "Ma et al. (AAAI 2024)"),
        (PSCCNetModel, PSCCNet, "pscc_net", "Liu et al. (IEEE TCSVT 2022)"),
        (TruForAdapter, TruForNet, "trufor", "Guillaro et al. (CVPR 2023)"),
        (CoDEModel, CoDENet, "code", "Peng et al. (IEEE TIFS 2024)"),
        (MaskCLIPModel, MaskCLIPNet, "maskclip", "Wang et al. (CVPR 2025)"),
        (SIDAModel, SIDANet, "sida", "Huang et al. (CVPR 2025)"),
        (RITAModel, RITANet, "rita", "Zhu et al. (CVPR Findings 2026)"),
    ],
)
def test_neural_models_quit_with_notice_without_checkpoint(
    model_cls, net_cls, expected_name, expected_ref, synthetic_image_and_mask
):
    """All neural models must quit with notice and refuse to run with random weights."""
    img, _ = synthetic_image_and_mask
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Without a checkpoint, model must quit with notice and block random init
        model_unavail = model_cls(
            device="cpu",
            cache_dir=tmpdir,
            auto_download=False,
            require_checkpoint=False,  # Even if False is passed, random init is strictly blocked
        )
        assert model_unavail.name == expected_name
        assert not model_unavail.is_available()
        info = model_unavail.get_model_info()
        assert info["available"] is False
        assert info["random_init_blocked"] is True
        assert info["status_notice"] is not None
        assert "Quitting with notice" in info["status_notice"]
        if expected_ref is not None:
            assert expected_ref in info["paper_reference"]

        with pytest.raises(RuntimeError, match="not available"):
            model_unavail.predict(img)

        # 2. With a valid checkpoint, model loads and predicts properly
        ckpt_path = _save_valid_checkpoint(net_cls, os.path.join(tmpdir, f"{expected_name}.pth"))
        model_loaded = model_cls(
            checkpoint_path=ckpt_path,
            device="cpu",
            auto_download=False,
        )
        assert model_loaded.is_available()
        loaded_info = model_loaded.get_model_info()
        assert loaded_info["available"] is True
        assert loaded_info["random_init_blocked"] is False
        assert loaded_info["checkpoint_path"] == ckpt_path

        out = model_loaded.predict(img)
        assert out.binary_mask.shape == (img.height, img.width)
        assert out.probability_map.shape == (img.height, img.width)
        assert 0.0 <= out.probability_map.min() <= out.probability_map.max() <= 1.0
        assert set(np.unique(out.binary_mask)).issubset({0, 1})


def test_mismatched_checkpoint_is_rejected_with_notice(synthetic_image_and_mask):
    """A checkpoint whose keys/shapes do not match the architecture must be rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_ckpt = os.path.join(tmpdir, "bad_weights.pth")
        torch.save({"state_dict": {"unrelated.weight": torch.randn(4, 4)}}, bad_ckpt)

        model = SparseViTModel(checkpoint_path=bad_ckpt, device="cpu", auto_download=False)
        assert not model.is_available()
        info = model.get_model_info()
        assert info["random_init_blocked"] is True
        assert "matched only 0/" in info["status_notice"]
        with pytest.raises(RuntimeError):
            model.predict(synthetic_image_and_mask[0])


def test_all_paper_models_gpu_with_checkpoints(synthetic_image_and_mask):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    img, _ = synthetic_image_and_mask
    paper_model_pairs = [
        (SparseViTModel, SparseViTNet),
        (IMLViTModel, IMLViTNet),
        (PSCCNetModel, PSCCNet),
        (TruForAdapter, TruForNet),
        (CoDEModel, CoDENet),
        (MaskCLIPModel, MaskCLIPNet),
        (SIDAModel, SIDANet),
        (RITAModel, RITANet),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for ModelCls, NetCls in paper_model_pairs:
            ckpt_path = _save_valid_checkpoint(NetCls, os.path.join(tmpdir, f"{ModelCls.__name__}.pth"))
            mod = ModelCls(checkpoint_path=ckpt_path, device="cuda", auto_download=False)
            assert mod.is_available()
            assert str(mod.device).startswith("cuda")
            out = mod.predict(img)
            assert out.binary_mask.shape == (img.height, img.width)
            assert out.inference_time_ms > 0


def test_official_pscc_net_and_code_architectures(synthetic_image_and_mask):
    """Verify official paper architectures (OfficialPSCCNet and OfficialCoDENet) load and run."""
    img, _ = synthetic_image_and_mask
    with tempfile.TemporaryDirectory() as tmpdir:
        pscc_off = OfficialPSCCNet()
        pscc_ckpt = os.path.join(tmpdir, "pscc_official.pth")
        torch.save(
            {
                "pscc_official_components": {
                    "FENet": pscc_off.FENet.state_dict(),
                    "SegNet": pscc_off.SegNet.state_dict(),
                    "ClsNet": pscc_off.ClsNet.state_dict(),
                }
            },
            pscc_ckpt,
        )
        pscc_mod = PSCCNetModel(checkpoint_path=pscc_ckpt, device="cpu", auto_download=False)
        assert pscc_mod.is_available()
        out_pscc = pscc_mod.predict(img)
        assert out_pscc.binary_mask.shape == (img.height, img.width)

        code_off = OfficialCoDENet(encoder="efficientnet-b0", steps=1)
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = code_off(x)
        assert y.shape == (1, 1, 64, 64)


def test_checkpoint_hub_registry_and_resolution():
    for name in ["pscc_net", "trufor", "iml_vit", "sparsevit", "code", "maskclip", "sida", "rita", "deep_forensic_detector"]:
        info = get_online_checkpoint_info(name)
        assert info["default_filename"].endswith((".pth", ".tar"))

    with tempfile.TemporaryDirectory() as tmpdir:
        path, notice = resolve_or_download_checkpoint(
            "rita",
            cache_dir=tmpdir,
            auto_download=True,
        )
        assert path is None
        assert notice is not None
        assert "Baidu Netdisk" in notice


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
    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs = {m: {"cache_dir": tmpdir, "auto_download": False} for m in expected_models}
        report = ModelRegistry.generate_availability_report(
            expected_models, model_kwargs=kwargs, default_device="cpu"
        )
        assert len(report) == 8
        for entry in report:
            assert entry["registered"] is True
            assert entry["load_successful"] is False
            assert entry["available"] is False
            assert entry["random_init_blocked"] is True


def test_require_checkpoint_flag():
    model = SparseViTModel(checkpoint_path="/nonexistent/path.pth", require_checkpoint=True, auto_download=False)
    assert not model.is_available()
    with pytest.raises(RuntimeError):
        model.predict(None)


def test_model_categories():
    assert "ela_detector" in ModelRegistry.list_classical_models()
    assert "trufor" in ModelRegistry.list_integrated_paper_models()
    assert ModelRegistry.get_model_category("ela_detector") == "Classical Models"
    assert ModelRegistry.get_model_category("noise_residual_detector") == "Classical Models"
    assert ModelRegistry.get_model_category("code") == "Integrated Models"
    assert ModelRegistry.get_model_category("trufor") == "Integrated Models"
    assert ModelRegistry.is_classical_model("ela_detector")
    assert ModelRegistry.is_integrated_model("code")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = ModelRegistry.generate_availability_report(
            ["ela_detector", "code"],
            model_kwargs={"code": {"cache_dir": tmpdir, "auto_download": False}},
            default_device="cpu",
        )
        assert report[0]["category"] == "Classical Models"
        assert report[1]["category"] == "Integrated Models"
