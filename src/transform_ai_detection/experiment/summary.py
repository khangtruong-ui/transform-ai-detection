"""Aggregate statistics, paired significance tests, failure analysis, and summary generator."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from transform_ai_detection.statistics.bootstrap import compute_image_level_bootstrap_ci
from transform_ai_detection.statistics.paired_tests import run_paired_significance_test
from transform_ai_detection.statistics.correlation import (
    compute_bivariate_correlation,
    compute_partial_correlation,
)
from transform_ai_detection.failure_analysis.shared import analyze_shared_failures
from transform_ai_detection.failure_analysis.stratification import (
    bin_mask_area,
    bin_resolution,
    stratify_performance,
)

logger = logging.getLogger(__name__)


class ExperimentSummaryGenerator:
    """Consolidates per-image metrics and diagnostics into publication-ready aggregate artifacts."""

    def __init__(
        self,
        metrics_csv_path: str,
        diagnostics_csv_path: str,
        output_dir: str,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        oom_models: Optional[List[str]] = None,
    ) -> None:
        self.metrics_csv_path = metrics_csv_path
        self.diagnostics_csv_path = diagnostics_csv_path
        self.output_dir = output_dir
        self.n_bootstraps = n_bootstraps
        self.ci = ci
        self.oom_models = list(oom_models or [])
        if not self.oom_models and os.path.exists(os.path.join(output_dir, "oom_models.json")):
            try:
                with open(os.path.join(output_dir, "oom_models.json"), "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.oom_models = loaded
            except Exception:
                pass

    def generate(self) -> Dict[str, Any]:
        """Generate all aggregate statistical tables and save to disk."""
        if not os.path.exists(self.metrics_csv_path):
            logger.warning(f"Metrics CSV not found at {self.metrics_csv_path}")
            return {}

        df_metrics = pd.read_csv(self.metrics_csv_path)
        if df_metrics.empty:
            logger.warning("Metrics CSV is empty.")
            return {}

        df_diag = pd.DataFrame()
        if os.path.exists(self.diagnostics_csv_path):
            df_diag = pd.read_csv(self.diagnostics_csv_path)

        # Merge on sample_id for identity baseline diagnostics
        if not df_diag.empty and "sample_id" in df_diag.columns:
            # Keep identity baseline diagnostics for property correlations
            identity_diag = df_diag[df_diag["transform_name"] == "identity"].drop_duplicates(
                subset=["sample_id"]
            )
            merged_df = pd.merge(df_metrics, identity_diag, on="sample_id", how="left", suffixes=("", "_diag"))
        else:
            merged_df = df_metrics.copy()

        # Add stratification bins if columns exist
        if "manipulation_area_fraction" in merged_df.columns:
            merged_df["mask_area_tier"] = merged_df["manipulation_area_fraction"].apply(bin_mask_area)
        elif "gt_area_fraction" in merged_df.columns:
            merged_df["mask_area_tier"] = merged_df["gt_area_fraction"].apply(bin_mask_area)

        if "pixel_count" in merged_df.columns:
            merged_df["resolution_tier"] = merged_df["pixel_count"].apply(bin_resolution)

        # 1. Aggregate metrics with bootstrap CIs
        agg_records = []
        metric_keys = [
            "iou",
            "f1",
            "precision",
            "recall",
            "boundary_f1",
            "auroc",
            "average_precision",
            "delta_iou",
            "delta_f1",
            "delta_boundary_f1",
        ]

        group_cols = ["model", "transform_name", "severity_label"]
        for group_keys, group in merged_df.groupby(group_cols):
            model, t_name, sev_label = group_keys
            rec: Dict[str, Any] = {
                "model": model,
                "transform_name": t_name,
                "severity_label": sev_label,
                "n_samples": len(group),
                "failure_rate": float(group["is_failure"].mean()) if "is_failure" in group.columns else 0.0,
            }

            for m_key in metric_keys:
                if m_key in group.columns:
                    vals = group[m_key].dropna().values
                    boot = compute_image_level_bootstrap_ci(
                        vals, n_bootstraps=self.n_bootstraps, ci=self.ci
                    )
                    rec[f"{m_key}_mean"] = boot["mean"]
                    rec[f"{m_key}_ci_lower"] = boot["ci_lower"]
                    rec[f"{m_key}_ci_upper"] = boot["ci_upper"]
                    rec[f"{m_key}_std_err"] = boot["std_err"]

            agg_records.append(rec)

        agg_df = pd.DataFrame(agg_records)
        agg_df.to_csv(os.path.join(self.output_dir, "aggregate_metrics.csv"), index=False)
        with open(os.path.join(self.output_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(agg_records, f, indent=2)

        # 2. Paired statistical significance tests (Transformed vs Identity baseline)
        paired_tests = []
        for model in merged_df["model"].unique():
            model_df = merged_df[merged_df["model"] == model]
            base_rows = model_df[model_df["transform_name"] == "identity"].set_index("sample_id")

            for (t_name, sev_label), trans_group in model_df.groupby(["transform_name", "severity_label"]):
                if t_name == "identity":
                    continue
                trans_rows = trans_group.set_index("sample_id")
                common_ids = base_rows.index.intersection(trans_rows.index)

                if len(common_ids) > 1:
                    b_iou = base_rows.loc[common_ids, "iou"].values
                    t_iou = trans_rows.loc[common_ids, "iou"].values
                    stat_res = run_paired_significance_test(b_iou, t_iou)
                    stat_res.update({
                        "model": model,
                        "transform_name": t_name,
                        "severity_label": sev_label,
                        "metric": "iou",
                    })
                    paired_tests.append(stat_res)

        with open(os.path.join(self.output_dir, "statistical_tests.json"), "w", encoding="utf-8") as f:
            json.dump(paired_tests, f, indent=2)

        # 3. Correlation between image diagnostics and degradation (Delta IoU)
        corr_records = []
        diag_features = [
            "manipulation_area_fraction",
            "isoperimetric_quotient",
            "convex_solidity",
            "min_border_distance_norm",
            "connected_components_count",
            "grad_magnitude_mean",
            "laplacian_variance",
            "shannon_entropy",
        ]

        for model in merged_df["model"].unique():
            m_df = merged_df[merged_df["model"] == model]
            for t_name in m_df["transform_name"].unique():
                if t_name == "identity":
                    continue
                sub_df = m_df[m_df["transform_name"] == t_name]
                if "delta_iou" not in sub_df.columns:
                    continue

                delta_y = sub_df["delta_iou"].values
                for feat in diag_features:
                    if feat in sub_df.columns:
                        x = sub_df[feat].values
                        bivar = compute_bivariate_correlation(x, delta_y)

                        # Partial correlation controlling for mask area and pixel count
                        covars = []
                        if "manipulation_area_fraction" in sub_df.columns and feat != "manipulation_area_fraction":
                            covars.append(sub_df["manipulation_area_fraction"].values)
                        if "pixel_count" in sub_df.columns and feat != "pixel_count":
                            covars.append(sub_df["pixel_count"].values)

                        partial_r = None
                        if len(covars) > 0:
                            Z = np.column_stack(covars)
                            part_res = compute_partial_correlation(x, delta_y, Z)
                            partial_r = part_res.get("pearson_r")

                        corr_records.append({
                            "model": model,
                            "transform_name": t_name,
                            "diagnostic_feature": feat,
                            "pearson_r": bivar.get("pearson_r"),
                            "pearson_p": bivar.get("pearson_p"),
                            "spearman_rho": bivar.get("spearman_rho"),
                            "spearman_p": bivar.get("spearman_p"),
                            "partial_pearson_r_controlled": partial_r,
                            "n_samples": bivar.get("n"),
                        })

        corr_df = pd.DataFrame(corr_records)
        corr_df.to_csv(os.path.join(self.output_dir, "correlations.csv"), index=False)

        # 4. Shared and model-specific failure analysis
        failure_analysis = analyze_shared_failures(merged_df, failure_col="is_failure")
        with open(os.path.join(self.output_dir, "failure_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, indent=2)

        # 5. Method and area stratification
        strat_method = stratify_performance(merged_df, group_col="method")
        strat_method.to_csv(os.path.join(self.output_dir, "stratification_by_method.csv"), index=False)

        strat_area = pd.DataFrame()
        if "mask_area_tier" in merged_df.columns:
            strat_area = stratify_performance(merged_df, group_col="mask_area_tier")
            strat_area.to_csv(os.path.join(self.output_dir, "stratification_by_mask_area.csv"), index=False)

        # 6. Overall Run Summary
        run_summary = {
            "total_evaluations": len(merged_df),
            "unique_samples": int(merged_df["sample_id"].nunique()),
            "models_evaluated": sorted(merged_df["model"].unique().tolist()) if "model" in merged_df.columns else [],
            "oom_models_skipped": self.oom_models,
            "transformations_evaluated": sorted(merged_df["transform_name"].unique().tolist()) if "transform_name" in merged_df.columns else [],
            "overall_failure_rate": float(merged_df["is_failure"].mean()) if "is_failure" in merged_df.columns else 0.0,
            "failure_analysis_summary": {
                "all_models_failed_rate": failure_analysis.get("all_models_failed_rate"),
                "model_specific_failure_count": failure_analysis.get("model_specific_failure_count"),
            },
            "artifacts_generated": [
                "aggregate_metrics.csv",
                "aggregate_metrics.json",
                "statistical_tests.json",
                "correlations.csv",
                "failure_analysis.json",
                "stratification_by_method.csv",
                "stratification_by_mask_area.csv",
            ] + (["oom_models.json"] if self.oom_models else []),
        }

        with open(os.path.join(self.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        return run_summary
