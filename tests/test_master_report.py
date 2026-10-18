"""Tests for Master Report Categorization and Tables."""

import json
import os
import tempfile
import pandas as pd
import pytest

from transform_ai_detection.experiment.master_report import generate_master_report


def test_master_report_categorizes_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create models report
        models_rep = [
            {"name": "ela_detector", "family": "forensic", "device": "cpu", "load_successful": True, "description": "ELA"},
            {"name": "dct_inconsistency_detector", "family": "freq", "device": "cpu", "load_successful": True, "description": "DCT"},
            {"name": "code", "family": "RL", "device": "cpu", "load_successful": True, "description": "CoDE"},
            {"name": "trufor", "family": "transformer", "device": "cpu", "load_successful": True, "description": "TruFor"},
        ]
        with open(os.path.join(tmpdir, "models_report.json"), "w") as f:
            json.dump(models_rep, f)

        # Create aggregate_metrics.csv
        agg_data = [
            # Classical
            {"model": "ela_detector", "transform_name": "identity", "severity_label": "original", "n_samples": 10, "iou_mean": 0.5, "iou_ci_lower": 0.4, "iou_ci_upper": 0.6, "f1_mean": 0.6, "f1_ci_lower": 0.5, "f1_ci_upper": 0.7, "delta_iou_mean": 0.0, "failure_rate": 0.1},
            {"model": "dct_inconsistency_detector", "transform_name": "identity", "severity_label": "original", "n_samples": 10, "iou_mean": 0.4, "iou_ci_lower": 0.3, "iou_ci_upper": 0.5, "f1_mean": 0.5, "f1_ci_lower": 0.4, "f1_ci_upper": 0.6, "delta_iou_mean": 0.0, "failure_rate": 0.2},
            # Integrated
            {"model": "code", "transform_name": "identity", "severity_label": "original", "n_samples": 10, "iou_mean": 0.7, "iou_ci_lower": 0.6, "iou_ci_upper": 0.8, "f1_mean": 0.8, "f1_ci_lower": 0.7, "f1_ci_upper": 0.9, "delta_iou_mean": 0.0, "failure_rate": 0.05},
            {"model": "trufor", "transform_name": "identity", "severity_label": "original", "n_samples": 10, "iou_mean": 0.8, "iou_ci_lower": 0.7, "iou_ci_upper": 0.9, "f1_mean": 0.85, "f1_ci_lower": 0.75, "f1_ci_upper": 0.95, "delta_iou_mean": 0.0, "failure_rate": 0.02},
        ]
        pd.DataFrame(agg_data).to_csv(os.path.join(tmpdir, "aggregate_metrics.csv"), index=False)

        # Create statistical_tests.json (both classical and integrated, multiple models)
        stat_tests = [
            {"model": "ela_detector", "transform_name": "jpeg_compression", "severity_label": "Q=50", "n_samples": 10, "mean_difference": -0.05, "wilcoxon_stat": 10.0, "wilcoxon_pvalue": 0.01, "cohens_d": -0.4, "is_significant_005": True},
            {"model": "dct_inconsistency_detector", "transform_name": "jpeg_compression", "severity_label": "Q=50", "n_samples": 10, "mean_difference": -0.02, "wilcoxon_stat": 12.0, "wilcoxon_pvalue": 0.04, "cohens_d": -0.2, "is_significant_005": True},
            {"model": "code", "transform_name": "jpeg_compression", "severity_label": "Q=50", "n_samples": 10, "mean_difference": -0.01, "wilcoxon_stat": 15.0, "wilcoxon_pvalue": 0.05, "cohens_d": -0.1, "is_significant_005": True},
            {"model": "trufor", "transform_name": "jpeg_compression", "severity_label": "Q=50", "n_samples": 10, "mean_difference": -0.005, "wilcoxon_stat": 18.0, "wilcoxon_pvalue": 0.02, "cohens_d": -0.05, "is_significant_005": True},
        ]
        with open(os.path.join(tmpdir, "statistical_tests.json"), "w") as f:
            json.dump(stat_tests, f)

        # Create correlations.csv
        corr_data = [
            {"model": "ela_detector", "transform_name": "jpeg_compression", "diagnostic_feature": "convex_solidity", "pearson_r": 0.2, "pearson_p": 0.01, "spearman_rho": 0.15, "partial_pearson_r_controlled": 0.18, "n_samples": 10},
            {"model": "code", "transform_name": "jpeg_compression", "diagnostic_feature": "convex_solidity", "pearson_r": 0.3, "pearson_p": 0.005, "spearman_rho": 0.25, "partial_pearson_r_controlled": 0.28, "n_samples": 10},
        ]
        pd.DataFrame(corr_data).to_csv(os.path.join(tmpdir, "correlations.csv"), index=False)

        # Create failure_analysis.json
        failure_data = {
            "all_models_failed_rate": 0.1,
            "model_specific_failure_count": 2,
            "model_specific_breakdown": {
                "ela_detector": 1,
                "code": 1,
            },
        }
        with open(os.path.join(tmpdir, "failure_analysis.json"), "w") as f:
            json.dump(failure_data, f)

        # Create stratification_by_method.csv
        method_data = [
            {"method": "splicing", "sample_count": 5, "failure_rate": 0.2, "iou_mean": 0.6, "iou_ci_lower": 0.5, "iou_ci_upper": 0.7, "delta_iou_mean": -0.05, "delta_iou_ci_lower": -0.08, "delta_iou_ci_upper": -0.02},
            {"method": "inpainting", "sample_count": 5, "failure_rate": 0.8, "iou_mean": 0.2, "iou_ci_lower": 0.1, "iou_ci_upper": 0.3, "delta_iou_mean": -0.15, "delta_iou_ci_lower": -0.20, "delta_iou_ci_upper": -0.10},
        ]
        pd.DataFrame(method_data).to_csv(os.path.join(tmpdir, "stratification_by_method.csv"), index=False)

        report_file = generate_master_report(tmpdir)
        with open(report_file, "r") as f:
            content = f.read()

        # Check section 4 categories
        assert "### Classical Models (Forensic Baselines)" in content
        assert "### Integrated Research Models" in content

        # Check section 6 aggregate performance categories
        assert "#### Classical Models" in content
        assert "#### Integrated Models" in content

        # Check section 6 statistical tests categories and presence of both models
        assert "`ela_detector`" in content
        assert "`code`" in content
        assert "`trufor`" in content
        assert "`dct_inconsistency_detector`" in content

        # Check section 7 correlations categories
        assert "### Classical Models: Diagnostic Predictors" in content
        assert "### Integrated Models: Diagnostic Predictors" in content

        # Check section 8 breakdown table
        assert "### Model-Specific Failure Breakdown by Category" in content
        assert "| **Classical Models** |" in content
        assert "| **Integrated Models** |" in content

        # Check attentive verdict callout alerts under tables
        assert "> [!IMPORTANT]" in content
        assert "> **H1 Performance Verdict**:" in content
        assert "> **Statistical Significance Verdict**:" in content
        assert "> **H2 Diagnostic Verdict**:" in content
        assert "> **H3 Shared Failure Verdict**:" in content
        assert "> **Subgroup Robustness Verdict**:" in content


def test_master_report_renders_random_init_blocked_notices():
    with tempfile.TemporaryDirectory() as tmpdir:
        models_rep = [
            {
                "name": "ela_detector",
                "family": "forensic",
                "device": "cpu",
                "load_successful": True,
                "description": "ELA",
            },
            {
                "name": "pscc_net",
                "family": "cnn",
                "device": "cpu",
                "load_successful": True,
                "checkpoint_path": "/root/.cache/iml_models/pscc_net.pth",
                "online_checkpoint_source": "https://github.com/proteus1991/PSCC-Net",
                "random_init_blocked": False,
                "description": "PSCC-Net",
            },
            {
                "name": "rita",
                "family": "transformer",
                "device": "cpu",
                "load_successful": False,
                "random_init_blocked": True,
                "online_checkpoint_source": "Unavailable publicly (Baidu Netdisk)",
                "status_notice": "Quitting with notice: Official RITA weights are on password-protected Baidu Netdisk.",
                "description": "RITA",
            },
        ]
        with open(os.path.join(tmpdir, "models_report.json"), "w") as f:
            json.dump(models_rep, f)

        report_file = generate_master_report(tmpdir)
        with open(report_file, "r") as f:
            content = f.read()

        assert "Strict Pretrained Checkpoint Policy (Random Initialization Prohibited)" in content
        assert "Quit with Notice (No Pretrained Weights — Random Init Prohibited)" in content
        assert "Active (Pretrained Checkpoint Loaded)" in content
        assert "### Online Checkpoint Discovery & Notice Log" in content
        assert "`rita`" in content
        assert "`pscc_net`" in content

