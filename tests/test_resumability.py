"""Unit tests for experiment ledger, checkpointing, and resumability."""

from __future__ import annotations

import os
import tempfile
import pytest

from transform_ai_detection.experiment.ledger import ExperimentLedger


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
