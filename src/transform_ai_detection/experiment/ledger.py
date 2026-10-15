"""Persistent results ledger and checkpointing for experiment resumability."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd


class ExperimentLedger:
    """Progressive, file-backed ledger recording per-sample experiment evaluations.

    Guarantees idempotency and resumability by loading existing records on startup
    and appending new results transactionally as they complete.
    """

    def __init__(self, metrics_csv_path: str, diagnostics_csv_path: str) -> None:
        self.metrics_csv_path = metrics_csv_path
        self.diagnostics_csv_path = diagnostics_csv_path

        os.makedirs(os.path.dirname(os.path.abspath(metrics_csv_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(diagnostics_csv_path)), exist_ok=True)

        self._completed_evals: Set[Tuple[str, str, str, str]] = set()
        self._diagnosed_samples: Set[Tuple[str, str, str]] = set()
        self._completed_models: Set[str] = set()
        self._baseline_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._metrics_fields: Optional[List[str]] = None
        self._diagnostics_fields: Optional[List[str]] = None

        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Read existing CSV files to discover completed evaluations."""
        if os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0:
            try:
                df = pd.read_csv(self.metrics_csv_path)
                self._metrics_fields = list(df.columns)
                req_cols = {"sample_id", "model", "transform_name", "severity_label"}
                if req_cols.issubset(df.columns):
                    for _, row in df.iterrows():
                        sid = str(row["sample_id"])
                        mod = str(row["model"])
                        t_name = str(row["transform_name"])
                        t_sev = str(row["severity_label"])
                        key = (sid, mod, t_name, t_sev)
                        self._completed_evals.add(key)
                        self._completed_models.add(mod)

                        # Cache baseline metrics for delta computation and skipping
                        if t_name == "identity" and t_sev == "original":
                            self._baseline_metrics[(sid, mod)] = row.to_dict()
            except Exception:
                pass

        if os.path.exists(self.diagnostics_csv_path) and os.path.getsize(self.diagnostics_csv_path) > 0:
            try:
                df_diag = pd.read_csv(self.diagnostics_csv_path)
                self._diagnostics_fields = list(df_diag.columns)
                req_diag = {"sample_id", "transform_name", "severity_label"}
                if req_diag.issubset(df_diag.columns):
                    for _, row in df_diag.iterrows():
                        key_d = (
                            str(row["sample_id"]),
                            str(row["transform_name"]),
                            str(row["severity_label"]),
                        )
                        self._diagnosed_samples.add(key_d)
            except Exception:
                pass

    def is_completed(
        self, sample_id: str, model_name: str, transform_name: str, severity_label: str
    ) -> bool:
        """Check if an evaluation has already been persisted to disk."""
        key = (str(sample_id), str(model_name), str(transform_name), str(severity_label))
        return key in self._completed_evals

    def get_completed_models(self) -> Set[str]:
        """Return set of all model names that have records in the ledger."""
        return set(self._completed_models)

    def get_baseline_metrics(self, sample_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Return cached baseline metrics for (sample_id, model_name) if available."""
        return self._baseline_metrics.get((str(sample_id), str(model_name)))

    def is_model_fully_completed(
        self,
        model_name: str,
        sample_ids: List[str],
        transform_keys: List[Tuple[str, str]],
    ) -> bool:
        """Check if every expected (sample_id, model_name, transform, severity) is completed."""
        if not sample_ids or not transform_keys:
            return False
        m = str(model_name)
        for sid in sample_ids:
            for t_name, t_sev in transform_keys:
                if (str(sid), m, str(t_name), str(t_sev)) not in self._completed_evals:
                    return False
        return True

    def get_model_progress(
        self,
        model_name: str,
        sample_ids: List[str],
        transform_keys: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Calculate completion progress for a given model."""
        m = str(model_name)
        completed = 0
        total = len(sample_ids) * len(transform_keys)
        for sid in sample_ids:
            for t_name, t_sev in transform_keys:
                if (str(sid), m, str(t_name), str(t_sev)) in self._completed_evals:
                    completed += 1
        return {
            "model": model_name,
            "completed": completed,
            "total": total,
            "is_complete": completed >= total and total > 0,
            "completion_ratio": completed / max(1, total),
        }

    def record_metric_row(self, row: Dict[str, Any]) -> None:
        """Append a completed evaluation metric row to disk."""
        sid = str(row["sample_id"])
        mod = str(row["model"])
        t_name = str(row["transform_name"])
        t_sev = str(row["severity_label"])

        key = (sid, mod, t_name, t_sev)
        self._completed_evals.add(key)
        self._completed_models.add(mod)

        if t_name == "identity" and t_sev == "original":
            self._baseline_metrics[(sid, mod)] = dict(row)

        file_exists = os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0
        if not file_exists or self._metrics_fields is None:
            self._metrics_fields = list(row.keys())
            with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
        else:
            new_keys = [k for k in row.keys() if k not in self._metrics_fields]
            if new_keys:
                df = pd.read_csv(self.metrics_csv_path)
                for k in new_keys:
                    df[k] = None
                self._metrics_fields.extend(new_keys)
                df.to_csv(self.metrics_csv_path, index=False)

            with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics_fields, extrasaction="ignore")
                writer.writerow(row)

    def record_diagnostic_row(self, row: Dict[str, Any]) -> None:
        """Append a pre-inference or transformed image diagnostic row to disk."""
        key = (
            str(row["sample_id"]),
            str(row["transform_name"]),
            str(row["severity_label"]),
        )
        if key in self._diagnosed_samples:
            return
        self._diagnosed_samples.add(key)

        file_exists = (
            os.path.exists(self.diagnostics_csv_path)
            and os.path.getsize(self.diagnostics_csv_path) > 0
        )
        if not file_exists or self._diagnostics_fields is None:
            self._diagnostics_fields = list(row.keys())
            with open(self.diagnostics_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._diagnostics_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
        else:
            new_keys = [k for k in row.keys() if k not in self._diagnostics_fields]
            if new_keys:
                df = pd.read_csv(self.diagnostics_csv_path)
                for k in new_keys:
                    df[k] = None
                self._diagnostics_fields.extend(new_keys)
                df.to_csv(self.diagnostics_csv_path, index=False)

            with open(self.diagnostics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._diagnostics_fields, extrasaction="ignore")
                writer.writerow(row)

    @property
    def completed_count(self) -> int:
        return len(self._completed_evals)
