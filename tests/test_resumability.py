"""Unit tests for experiment ledger, checkpointing, and resumability."""

from __future__ import annotations

import os
import tempfile
import pytest

from transform_ai_detection.config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    TransformationConfig,
)
from transform_ai_detection.experiment.ledger import ExperimentLedger
from transform_ai_detection.experiment.runner import ExperimentRunner


def test_ledger_persistence_and_resumability():
    with tempfile.TemporaryDirectory() as tmpdir:
        m_csv = os.path.join(tmpdir, "metrics.csv")
        d_csv = os.path.join(tmpdir, "diag.csv")

        # 1. Initialize ledger and record rows
        ledger = ExperimentLedger(m_csv, d_csv)
        assert ledger.completed_count == 0

        ledger.record_metric_row({
            "sample_id": "s1",
            "model": "m1",
            "transform_name": "jpeg",
            "severity_label": "Q=50",
            "iou": 0.65,
        })
        ledger.record_metric_row({
            "sample_id": "s1",
            "model": "m2",
            "transform_name": "jpeg",
            "severity_label": "Q=50",
            "iou": 0.45,
        })
        assert ledger.completed_count == 2
        assert ledger.is_completed("s1", "m1", "jpeg", "Q=50")
        assert not ledger.is_completed("s1", "m1", "blur", "sigma=1.0")

        # 2. Re-instantiate ledger (simulating experiment restart)
        resumed_ledger = ExperimentLedger(m_csv, d_csv)
        assert resumed_ledger.completed_count == 2
        assert resumed_ledger.is_completed("s1", "m1", "jpeg", "Q=50")
        assert resumed_ledger.is_completed("s1", "m2", "jpeg", "Q=50")

        # 3. Add new entry
        resumed_ledger.record_metric_row({
            "sample_id": "s2",
            "model": "m1",
            "transform_name": "jpeg",
            "severity_label": "Q=50",
            "iou": 0.80,
        })
        assert resumed_ledger.completed_count == 3


def test_ledger_existing_models_detection_and_completion():
    with tempfile.TemporaryDirectory() as tmpdir:
        m_csv = os.path.join(tmpdir, "metrics.csv")
        d_csv = os.path.join(tmpdir, "diag.csv")

        ledger = ExperimentLedger(m_csv, d_csv)
        sample_ids = ["sample_0", "sample_1"]
        transforms = [("identity", "original"), ("jpeg_compression", "Q=75")]

        assert ledger.get_completed_models() == set()
        assert not ledger.is_model_fully_completed("sparsevit", sample_ids, transforms)

        # Record all evaluations for model sparsevit
        for sid in sample_ids:
            for t_name, t_sev in transforms:
                ledger.record_metric_row({
                    "sample_id": sid,
                    "model": "sparsevit",
                    "transform_name": t_name,
                    "severity_label": t_sev,
                    "iou": 0.85,
                })

        assert ledger.get_completed_models() == {"sparsevit"}
        assert ledger.is_model_fully_completed("sparsevit", sample_ids, transforms)
        assert not ledger.is_model_fully_completed("iml_vit", sample_ids, transforms)

        prog = ledger.get_model_progress("sparsevit", sample_ids, transforms)
        assert prog["is_complete"] is True
        assert prog["completed"] == 4
        assert prog["total"] == 4


def test_ledger_baseline_caching_and_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        m_csv = os.path.join(tmpdir, "metrics.csv")
        d_csv = os.path.join(tmpdir, "diag.csv")

        ledger = ExperimentLedger(m_csv, d_csv)
        row = {
            "sample_id": "sample_42",
            "model": "code",
            "transform_name": "identity",
            "severity_label": "original",
            "iou": 0.77,
            "f1": 0.88,
        }
        ledger.record_metric_row(row)

        cached = ledger.get_baseline_metrics("sample_42", "code")
        assert cached is not None
        assert cached["iou"] == 0.77
        assert cached["f1"] == 0.88

        # Reload from disk and verify cache is populated
        resumed = ExperimentLedger(m_csv, d_csv)
        cached_resumed = resumed.get_baseline_metrics("sample_42", "code")
        assert cached_resumed is not None
        assert cached_resumed["iou"] == 0.77


def test_runner_detect_existing_models_and_skip():
    """Verify that ExperimentRunner detects previously completed models and skips them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            name="test_skip_existing",
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
                device="cpu",
                skip_existing_models=True,
            ),
        )
        config.output.output_dir = tmpdir
        config.output.save_visualizations = False

        # First run: both mock_zeros and mock_ones evaluate (2 samples * 2 models = 4)
        runner1 = ExperimentRunner(config)
        summary1 = runner1.run()
        assert summary1["total_evaluations_completed"] == 4

        # Second run with same config + skip_existing_models=True
        # Both models are already completed and should be detected and skipped
        runner2 = ExperimentRunner(config)
        summary2 = runner2.run()
        # No new evaluations performed, total stays 4
        assert summary2["total_evaluations_completed"] == 4


def test_skip_existing_models_defaults_to_true():
    """Verify ModelConfig.skip_existing_models defaults to True."""
    cfg = ModelConfig()
    assert cfg.skip_existing_models is True


def test_ledger_fast_path_model_completion_without_sample_ids():
    """Verify is_model_fully_completed fast in-memory check without explicit sample IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m_csv = os.path.join(tmpdir, "metrics.csv")
        d_csv = os.path.join(tmpdir, "diag.csv")

        ledger = ExperimentLedger(m_csv, d_csv)
        transforms = [("identity", "original"), ("jpeg_compression", "Q=75")]

        assert not ledger.is_model_fully_completed(
            "sparsevit",
            sample_ids=None,
            transform_keys=transforms,
            expected_sample_count=2,
        )

        for sid in ["s1", "s2"]:
            for t_name, t_sev in transforms:
                ledger.record_metric_row({
                    "sample_id": sid,
                    "model": "sparsevit",
                    "transform_name": t_name,
                    "severity_label": t_sev,
                })

        # Now sparsevit has 2 samples * 2 transforms = 4 evaluations
        assert ledger.get_model_evaluation_count("sparsevit") == 4
        assert ledger.is_model_fully_completed(
            "sparsevit",
            sample_ids=None,
            transform_keys=transforms,
            expected_sample_count=2,
        )
        # But if we expected 3 samples, it is not completed
        assert not ledger.is_model_fully_completed(
            "sparsevit",
            sample_ids=None,
            transform_keys=transforms,
            expected_sample_count=3,
        )
