"""Main experiment runner with resumability, graceful OOM recovery, and paired baseline execution."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

from transform_ai_detection.config import ExperimentConfig
from transform_ai_detection.hardware import detect_hardware, get_torch_device
from transform_ai_detection.dataset.base import BaseDatasetAdapter, DatasetSample
from transform_ai_detection.dataset.validator import DatasetValidator
from transform_ai_detection.dataset.tampered_combined import TamperedCombinedDatasetAdapter
from transform_ai_detection.dataset.synthetic import SyntheticDatasetAdapter
from transform_ai_detection.transformations.base import (
    BaseTransformation,
    TransformationRegistry,
    TransformResult,
)
from transform_ai_detection.transformations.classical import (
    IdentityTransformation,
    JPEGCompressionTransformation,
    GaussianBlurTransformation,
    DownscaleUpscaleTransformation,
    BrightnessContrastTransformation,
)
from transform_ai_detection.transformations.noise_sharp import (
    GaussianNoiseTransformation,
    SharpeningTransformation,
)
from transform_ai_detection.transformations.geometric import (
    HorizontalFlipTransformation,
    VerticalFlipTransformation,
    CenterCropTransformation,
)
from transform_ai_detection.models.base import BaseLocalizationModel, ModelOutput
from transform_ai_detection.models.registry import ModelRegistry
from transform_ai_detection.metrics.engine import EvaluationEngine
from transform_ai_detection.diagnostics.geometry import compute_geometric_diagnostics
from transform_ai_detection.diagnostics.texture import compute_texture_diagnostics
from transform_ai_detection.diagnostics.error_diag import compute_post_inference_diagnostics
from transform_ai_detection.failure_analysis.classifier import FailureClassifier
from transform_ai_detection.experiment.ledger import ExperimentLedger
from transform_ai_detection.experiment.summary import ExperimentSummaryGenerator
from transform_ai_detection.experiment.master_report import generate_master_report

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Executes deterministic, paired, inference-only robustness experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = os.path.abspath(config.output.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.preds_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(self.plots_dir, exist_ok=True)
        if config.output.save_predictions:
            os.makedirs(self.preds_dir, exist_ok=True)

        # 1. Setup persistent files
        self.metrics_csv = os.path.join(self.output_dir, "metrics_per_image.csv")
        self.diagnostics_csv = os.path.join(self.output_dir, "diagnostics_per_image.csv")
        self.ledger = ExperimentLedger(self.metrics_csv, self.diagnostics_csv)
        self.validator = DatasetValidator()

        # 2. Setup evaluation tools
        self.eval_engine = EvaluationEngine(
            boundary_tolerance=config.evaluation.boundary_tolerance,
            compute_ranking=config.evaluation.compute_ranking_metrics,
        )
        self.failure_classifier = FailureClassifier(
            failure_metric=config.evaluation.failure_metric,
            failure_threshold=config.evaluation.failure_threshold,
            degradation_delta_threshold=config.evaluation.degradation_failure_delta,
        )

        self.errors_log: List[Dict[str, Any]] = []

    def _init_environment_and_manifest(self, dataset: BaseDatasetAdapter, sample_indices: List[int]) -> None:
        """Persist environment, configs, dataset manifest, and model/transform metadata."""
        # Save config
        self.config.save_yaml(os.path.join(self.output_dir, "config.yaml"))

        # Save environment & hardware detection
        env_info = detect_hardware()
        with open(os.path.join(self.output_dir, "environment.json"), "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

        # Save models availability report
        models_report = ModelRegistry.generate_availability_report(
            self.config.models.models,
            self.config.models.model_kwargs,
            default_device=self.config.models.device,
        )
        with open(os.path.join(self.output_dir, "models_report.json"), "w", encoding="utf-8") as f:
            json.dump(models_report, f, indent=2)

        # Save dataset manifest
        manifest = {
            "dataset_name": self.config.dataset.name,
            "adapter": self.config.dataset.adapter,
            "split": self.config.dataset.split,
            "total_dataset_rows": len(dataset),
            "selected_sample_count": len(sample_indices),
            "sample_indices": sample_indices,
            "seed": self.config.dataset.seed,
        }
        with open(os.path.join(self.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _instantiate_dataset(self) -> BaseDatasetAdapter:
        """Instantiate dataset adapter specified in configuration."""
        cfg = self.config.dataset
        if cfg.adapter == "synthetic":
            return SyntheticDatasetAdapter(
                num_samples=cfg.max_samples or 20,
                seed=cfg.seed,
                validator=self.validator,
            )
        elif cfg.adapter == "tampered_combined":
            return TamperedCombinedDatasetAdapter(
                repo_id=cfg.name,
                split=cfg.split,
                streaming=cfg.streaming,
                max_cached_row_groups=cfg.max_cached_row_groups,
                validator=self.validator,
            )
        else:
            raise ValueError(f"Unsupported dataset adapter: {cfg.adapter}")

    def _build_transformations(self) -> List[BaseTransformation]:
        """Build list of transformation instances based on configuration."""
        transforms: List[BaseTransformation] = []
        enabled = self.config.transformations.enabled_transforms
        severities = self.config.transformations.severities

        # Always start with identity baseline
        transforms.append(IdentityTransformation(severity_level=0))

        for name in enabled:
            if name == "identity":
                continue
            sev_list = severities.get(name, [])
            for sev in sev_list:
                if name == "jpeg_compression":
                    transforms.append(JPEGCompressionTransformation(quality=sev))
                elif name == "gaussian_blur":
                    transforms.append(GaussianBlurTransformation(sigma=sev))
                elif name == "downscale_upscale":
                    transforms.append(DownscaleUpscaleTransformation(scale_factor=sev))
                elif name == "brightness_contrast":
                    if isinstance(sev, dict):
                        transforms.append(
                            BrightnessContrastTransformation(
                                factor_brightness=sev.get("factor_brightness", 1.0),
                                factor_contrast=sev.get("factor_contrast", 1.0),
                            )
                        )
                elif name == "gaussian_noise":
                    transforms.append(GaussianNoiseTransformation(std=sev))
                elif name == "sharpening":
                    transforms.append(SharpeningTransformation(factor=sev))
                elif name == "horizontal_flip":
                    transforms.append(HorizontalFlipTransformation())
                elif name == "vertical_flip":
                    transforms.append(VerticalFlipTransformation())
                elif name == "center_crop":
                    transforms.append(CenterCropTransformation(crop_ratio=sev))

        # Save transformation metadata report
        trans_meta = [t.get_metadata().to_dict() for t in transforms]
        with open(os.path.join(self.output_dir, "transformations_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(trans_meta, f, indent=2)

        return transforms

    def run(self) -> Dict[str, Any]:
        """Execute full experiment run."""
        logger.info(f"Starting experiment: {self.config.name}")
        start_time = time.time()

        dataset = self._instantiate_dataset()
        sample_indices = dataset.deterministic_sample_indices(
            count=self.config.dataset.max_samples,
            seed=self.config.dataset.seed,
            indices=self.config.dataset.sample_indices,
        )

        self._init_environment_and_manifest(dataset, sample_indices)
        transformations = self._build_transformations()

        transform_keys = [(t.get_metadata().name, t.get_metadata().severity_label) for t in transformations]
        skip_existing = getattr(self.config.models, "skip_existing_models", True)

        existing_models = self.ledger.get_completed_models()
        if existing_models:
            logger.info(f"Discovered existing models in ledger: {sorted(list(existing_models))}")

        # Only fetch sample IDs upfront if the dataset supports instant / non-network access
        sample_ids_str: Optional[List[str]] = None
        if getattr(dataset, "can_instant_sample_ids", False):
            sample_ids_str = [dataset.get_sample_id(i) for i in sample_indices]

        # Instantiate active models
        models: Dict[str, BaseLocalizationModel] = {}
        target_device = self.config.models.device
        for m_name in self.config.models.models:
            if skip_existing and self.ledger.is_model_fully_completed(
                m_name,
                sample_ids=sample_ids_str,
                transform_keys=transform_keys,
                expected_sample_count=len(sample_indices),
            ):
                logger.info(f"Model '{m_name}' is already fully completed in {self.output_dir}. Skipping model initialization.")
                continue

            m_kw = dict(self.config.models.model_kwargs.get(m_name, {}))
            if "device" not in m_kw:
                m_kw["device"] = target_device
            try:
                mod = ModelRegistry.create(m_name, **m_kw)
                if mod.is_available():
                    models[m_name] = mod
                else:
                    logger.warning(f"Model {m_name} reported unavailable; skipping.")
            except Exception as e:
                logger.error(f"Failed to instantiate model {m_name}: {e}")
                self.errors_log.append({"stage": "model_init", "model": m_name, "error": str(e)})

        logger.info(
            f"Active models: {list(models.keys())} | "
            f"Transformations: {len(transformations)} | "
            f"Samples: {len(sample_indices)}"
        )

        active_indices = sample_indices if models else []
        if not models:
            logger.info("All configured models are already fully evaluated in the ledger. Skipping evaluation loop.")

        # Main evaluation loop over samples
        pbar = tqdm(active_indices, desc="Evaluating samples", dynamic_ncols=True)
        for idx in pbar:
            pbar.set_postfix(sample_idx=idx, refresh=False)
            try:
                sample: DatasetSample = dataset[idx]
            except Exception as e:
                logger.error(f"Failed to fetch sample index {idx}: {e}")
                self.errors_log.append({"stage": "sample_fetch", "index": idx, "error": str(e)})
                continue

            # Check sample validity
            if sample.metadata.get("is_valid") is False:
                logger.debug(f"Skipping invalid sample {sample.sample_id}")
                continue

            # Compute pre-inference diagnostics for baseline sample
            geo_diag = compute_geometric_diagnostics(sample.width, sample.height, sample.mask)
            tex_diag = compute_texture_diagnostics(sample.image)

            # Record baseline diagnostics to ledger
            diag_base_row = {
                "sample_id": sample.sample_id,
                "transform_name": "identity",
                "severity_label": "original",
                "label": sample.label,
                "label_text": sample.label_text,
                "method": sample.method,
                "dataset_source": sample.dataset_source,
                **geo_diag.to_dict(),
                **tex_diag.to_dict(),
            }
            self.ledger.record_diagnostic_row(diag_base_row)

            # Step A: Evaluate baseline (Identity) for each model
            baseline_metrics: Dict[str, Dict[str, Any]] = {}
            for m_name, model in models.items():
                if self.ledger.is_completed(sample.sample_id, m_name, "identity", "original"):
                    cached_base = self.ledger.get_baseline_metrics(sample.sample_id, m_name)
                    if cached_base is not None:
                        logger.debug(f"Skipping baseline for model {m_name} on sample {sample.sample_id} (already completed)")
                        baseline_metrics[m_name] = cached_base
                        continue

                try:
                    out: ModelOutput = model.predict(sample.image)
                    metrics = self.eval_engine.evaluate_sample(
                        y_true=sample.mask,
                        y_pred_binary=out.binary_mask,
                        y_pred_prob=out.probability_map,
                        seed=self.config.dataset.seed,
                    )
                    fail_info = self.failure_classifier.classify(metrics)
                    post_diag = compute_post_inference_diagnostics(
                        image=sample.image,
                        y_true=sample.mask,
                        y_pred=out.binary_mask,
                    )

                    base_row = {
                        "sample_id": sample.sample_id,
                        "model": m_name,
                        "transform_name": "identity",
                        "severity_label": "original",
                        "severity_level": "0",
                        "label": sample.label,
                        "label_text": sample.label_text,
                        "method": sample.method,
                        "dataset_source": sample.dataset_source,
                        "is_valid_comparison": True,
                        "inference_time_ms": out.inference_time_ms,
                        "is_failure": fail_info.is_failure,
                        "is_severe_degradation": False,
                        "is_transformation_induced_failure": False,
                        "delta_iou": 0.0,
                        "delta_f1": 0.0,
                        "delta_boundary_f1": 0.0,
                        "delta_precision": 0.0,
                        "delta_recall": 0.0,
                        "delta_fpr": 0.0,
                        "delta_fnr": 0.0,
                        "delta_auroc": 0.0,
                        "delta_average_precision": 0.0,
                        **metrics,
                        **post_diag.to_dict(),
                    }
                    self.ledger.record_metric_row(base_row)
                    baseline_metrics[m_name] = metrics

                except (torch.cuda.OutOfMemoryError, MemoryError) as oom:
                    logger.warning(f"OOM on sample {sample.sample_id}, model {m_name}: {oom}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.errors_log.append({
                        "stage": "inference_baseline_oom",
                        "sample_id": sample.sample_id,
                        "model": m_name,
                        "error": str(oom),
                    })
                except Exception as e:
                    logger.warning(f"Error on sample {sample.sample_id}, model {m_name}: {e}")
                    self.errors_log.append({
                        "stage": "inference_baseline_error",
                        "sample_id": sample.sample_id,
                        "model": m_name,
                        "error": str(e),
                    })

            # Step B: Evaluate all transformations under paired condition
            for transform in transformations:
                t_meta = transform.get_metadata()
                if t_meta.name == "identity":
                    continue

                # Apply transformation
                try:
                    t_res: TransformResult = transform.apply(
                        image=sample.image,
                        mask=sample.mask,
                        seed=self.config.transformations.seed,
                        sample_id=sample.sample_id,
                    )
                except Exception as e:
                    logger.warning(f"Transform {t_meta.name} failed on {sample.sample_id}: {e}")
                    continue

                if not t_res.is_valid_comparison:
                    logger.debug(
                        f"Transform {t_meta.name} invalidated comparison for {sample.sample_id}: "
                        f"{t_res.invalidation_reason}"
                    )
                    continue

                # Evaluate models on transformed image
                for m_name, model in models.items():
                    if self.ledger.is_completed(
                        sample.sample_id, m_name, t_meta.name, t_meta.severity_label
                    ):
                        continue

                    try:
                        out = model.predict(t_res.image)
                        trans_metrics = self.eval_engine.evaluate_sample(
                            y_true=t_res.mask,
                            y_pred_binary=out.binary_mask,
                            y_pred_prob=out.probability_map,
                            seed=self.config.dataset.seed,
                        )

                        base_m = baseline_metrics.get(m_name)
                        deltas = self.eval_engine.compute_deltas(base_m or {}, trans_metrics)
                        fail_info = self.failure_classifier.classify(
                            trans_metrics, baseline_metrics=base_m
                        )

                        # Post inference diagnostics
                        post_diag = compute_post_inference_diagnostics(
                            image=t_res.image,
                            y_true=t_res.mask,
                            y_pred=out.binary_mask,
                        )

                        row = {
                            "sample_id": sample.sample_id,
                            "model": m_name,
                            "transform_name": t_meta.name,
                            "severity_label": t_meta.severity_label,
                            "severity_level": str(t_meta.severity_level),
                            "label": sample.label,
                            "label_text": sample.label_text,
                            "method": sample.method,
                            "dataset_source": sample.dataset_source,
                            "is_valid_comparison": True,
                            "inference_time_ms": out.inference_time_ms,
                            "is_failure": fail_info.is_failure,
                            "is_severe_degradation": fail_info.is_severe_degradation,
                            "is_transformation_induced_failure": fail_info.is_transformation_induced_failure,
                            **deltas,
                            **trans_metrics,
                            **post_diag.to_dict(),
                        }
                        self.ledger.record_metric_row(row)

                    except (torch.cuda.OutOfMemoryError, MemoryError) as oom:
                        logger.warning(
                            f"OOM on {sample.sample_id}, model {m_name}, transform {t_meta.name}: {oom}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.errors_log.append({
                            "stage": "inference_transform_oom",
                            "sample_id": sample.sample_id,
                            "model": m_name,
                            "transform": t_meta.name,
                            "error": str(oom),
                        })
                    except Exception as e:
                        logger.warning(
                            f"Error on {sample.sample_id}, model {m_name}, transform {t_meta.name}: {e}"
                        )
                        self.errors_log.append({
                            "stage": "inference_transform_error",
                            "sample_id": sample.sample_id,
                            "model": m_name,
                            "transform": t_meta.name,
                            "error": str(e),
                        })

        # Save exclusion logger report and errors
        self.validator.exclusion_logger.export_json(
            os.path.join(self.output_dir, "dataset_exclusions.json")
        )
        with open(os.path.join(self.output_dir, "errors_and_skipped.json"), "w", encoding="utf-8") as f:
            json.dump(self.errors_log, f, indent=2)

        # Generate final aggregate metrics, stats, and run summary
        logger.info("Computing aggregate metrics, bootstrap CIs, and statistical tests...")
        summary_gen = ExperimentSummaryGenerator(
            metrics_csv_path=self.metrics_csv,
            diagnostics_csv_path=self.diagnostics_csv,
            output_dir=self.output_dir,
            n_bootstraps=self.config.evaluation.bootstrap_iterations,
            ci=self.config.evaluation.bootstrap_ci,
        )
        summary = summary_gen.generate()
        summary["duration_seconds"] = round(time.time() - start_time, 2)
        summary["total_evaluations_completed"] = self.ledger.completed_count

        # Generate Master Markdown Research Report
        logger.info("Generating master research report in Markdown...")
        report_file = generate_master_report(self.output_dir)
        summary["master_report_path"] = report_file

        logger.info(f"Experiment {self.config.name} finished in {summary['duration_seconds']}s.")
        logger.info(f"Master Markdown Report saved to: {report_file}")
        return summary
