"""Tests for skipping OOM and explicitly skipped models."""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
from PIL import Image

from transform_ai_detection.config import (
    ExperimentConfig,
    DatasetConfig,
    TransformationConfig,
    ModelConfig,
)
from transform_ai_detection.experiment.runner import ExperimentRunner, is_oom_exception
from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.models.registry import ModelRegistry


class MockOOMModel(BaseLocalizationModel):
    """Mock model that raises torch.cuda.OutOfMemoryError or MemoryError on predict."""

    call_count = 0

    def __init__(self, name: str = "mock_oom", fail_on_call: int = 1, device: str = "cpu", **kwargs) -> None:
        self._name = name
        self.device = device
        self.fail_on_call = fail_on_call

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        return {"name": self._name}

    def predict(self, image: Image.Image) -> ModelOutput:
        MockOOMModel.call_count += 1
        if MockOOMModel.call_count >= self.fail_on_call:
            raise MemoryError("CUDA out of memory. Tried to allocate 8.00 GiB")
        w, h = image.size
        return ModelOutput(
            binary_mask=np.zeros((h, w), dtype=np.uint8),
            probability_map=np.zeros((h, w), dtype=np.float32),
            inference_time_ms=1.0,
        )


def test_is_oom_exception_detection():
    assert is_oom_exception(MemoryError("CUDA out of memory"))
    assert is_oom_exception(RuntimeError("CUDA error: out of memory"))
    assert is_oom_exception(RuntimeError("RuntimeError: out of memory"))
    assert not is_oom_exception(ValueError("Invalid argument"))
    assert not is_oom_exception(RuntimeError("Shape mismatch"))


def test_skip_oom_models_config_defaults():
    cfg = ModelConfig()
    assert cfg.skip_oom_models is True
    assert cfg.skip_models == []


def test_dynamic_oom_skipping_during_baseline():
    """Verify that a model raising OOM during baseline is evicted and skipped for subsequent samples and transforms."""
    ModelRegistry.register("mock_oom_baseline", lambda **kw: MockOOMModel(name="mock_oom_baseline", fail_on_call=1, **kw))
    MockOOMModel.call_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_oom_baseline",
            dataset=DatasetConfig(
                adapter="synthetic",
                max_samples=3,
                seed=42,
            ),
            transformations=TransformationConfig(
                enabled_transforms=["identity", "jpeg_compression"],
                severities={"jpeg_compression": [75]},
            ),
            models=ModelConfig(
                models=["mock_zeros", "mock_oom_baseline"],
                device="cpu",
                skip_oom_models=True,
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        runner = ExperimentRunner(config)
        summary = runner.run()

        # mock_oom_baseline failed on call 1 (sample 0 baseline), so it should be skipped for sample 0 transforms and samples 1, 2!
        # mock_zeros should evaluate: 3 samples * 2 transforms (identity + jpeg) = 6 evaluations
        assert summary["total_evaluations_completed"] == 6
        assert "mock_oom_baseline" in summary["oom_models"]

        # oom_models.json must be persisted on disk
        oom_file = os.path.join(tmpdir, "oom_models.json")
        assert os.path.exists(oom_file)
        with open(oom_file, "r") as f:
            oom_data = json.load(f)
        assert "mock_oom_baseline" in oom_data

        # Resuming with skip_oom_models=True: mock_oom_baseline is not even initialized!
        runner2 = ExperimentRunner(config)
        assert "mock_oom_baseline" in runner2.oom_models
        summary2 = runner2.run()
        # Since mock_zeros is already completed, total stays 6
        assert summary2["total_evaluations_completed"] == 6


def test_dynamic_oom_skipping_during_transform():
    """Verify that a model raising OOM during transformation is evicted and skipped for subsequent evals."""
    # Sinks baseline for sample 0 (call 1), then fails on transform 1 (call 2)
    ModelRegistry.register("mock_oom_trans", lambda **kw: MockOOMModel(name="mock_oom_trans", fail_on_call=2, **kw))
    MockOOMModel.call_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_oom_trans",
            dataset=DatasetConfig(
                adapter="synthetic",
                max_samples=2,
                seed=42,
            ),
            transformations=TransformationConfig(
                enabled_transforms=["identity", "jpeg_compression"],
                severities={"jpeg_compression": [75]},
            ),
            models=ModelConfig(
                models=["mock_zeros", "mock_oom_trans"],
                device="cpu",
                skip_oom_models=True,
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        runner = ExperimentRunner(config)
        summary = runner.run()

        # mock_zeros: 2 samples * 2 transforms = 4
        # mock_oom_trans: sample 0 baseline completed (1), then OOM on transform (evicted)
        # Total evaluations = 5
        assert summary["total_evaluations_completed"] == 5
        assert "mock_oom_trans" in summary["oom_models"]
        assert os.path.exists(os.path.join(tmpdir, "oom_models.json"))


def test_explicit_skip_models_configuration():
    """Verify that models in skip_models are bypassed during model initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_skip_models",
            dataset=DatasetConfig(
                adapter="synthetic",
                max_samples=2,
                seed=42,
            ),
            transformations=TransformationConfig(
                enabled_transforms=["identity"],
                severities={},
            ),
            models=ModelConfig(
                models=["mock_zeros", "mock_ones"],
                skip_models=["mock_ones"],
                device="cpu",
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        runner = ExperimentRunner(config)
        summary = runner.run()

        # Only mock_zeros evaluates: 2 samples * 1 transform = 2 evaluations
        assert summary["total_evaluations_completed"] == 2


def test_all_models_oom_terminates_gracefully():
    """Verify that if all active models OOM, the runner terminates gracefully."""
    ModelRegistry.register("mock_oom_only", lambda **kw: MockOOMModel(name="mock_oom_only", fail_on_call=1, **kw))
    MockOOMModel.call_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_all_oom",
            dataset=DatasetConfig(
                adapter="synthetic",
                max_samples=3,
                seed=42,
            ),
            transformations=TransformationConfig(
                enabled_transforms=["identity"],
                severities={},
            ),
            models=ModelConfig(
                models=["mock_oom_only"],
                device="cpu",
                skip_oom_models=True,
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        runner = ExperimentRunner(config)
        summary = runner.run()

        assert summary["total_evaluations_completed"] == 0
        assert "mock_oom_only" in summary["oom_models"]
        assert os.path.exists(os.path.join(tmpdir, "oom_models.json"))


def test_instantiation_oom_is_skipped():
    """Verify that a model throwing OOM during instantiation is skipped and recorded."""
    class MockInitOOMModel:
        def __init__(self, **kw):
            raise MemoryError("CUDA out of memory during model weight allocation")

    ModelRegistry.register("mock_init_oom", lambda **kw: MockInitOOMModel(**kw))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_init_oom",
            dataset=DatasetConfig(
                adapter="synthetic",
                max_samples=2,
                seed=42,
            ),
            transformations=TransformationConfig(
                enabled_transforms=["identity"],
                severities={},
            ),
            models=ModelConfig(
                models=["mock_zeros", "mock_init_oom"],
                device="cpu",
                skip_oom_models=True,
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        runner = ExperimentRunner(config)
        summary = runner.run()

        assert summary["total_evaluations_completed"] == 2
        assert "mock_init_oom" in summary["oom_models"]


def test_master_report_displays_oom_models():
    """Verify master report includes warning and skipped status for OOM models."""
    from transform_ai_detection.experiment.master_report import generate_master_report

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write dummy oom_models.json and models_report.json
        with open(os.path.join(tmpdir, "oom_models.json"), "w") as f:
            json.dump(["sparsevit", "trufor"], f)

        with open(os.path.join(tmpdir, "models_report.json"), "w") as f:
            json.dump([
                {"name": "sparsevit", "family": "Transformer", "device": "cuda", "load_successful": True, "description": "SparseViT"},
                {"name": "ela_detector", "family": "Classical", "device": "cpu", "load_successful": True, "description": "ELA"},
            ], f)

        report_file = generate_master_report(tmpdir)
        with open(report_file, "r") as f:
            content = f.read()

        assert "GPU Out-of-Memory (OOM) Exclusions" in content
        assert "sparsevit" in content
        assert "Skipped (OOM - Out of Memory)" in content
