"""Command-Line Interface (CLI) for IML Robustness and Failure Analysis."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

from transform_ai_detection.config import ExperimentConfig, DatasetConfig, ModelConfig, TransformationConfig
from transform_ai_detection.experiment.runner import ExperimentRunner
from transform_ai_detection.experiment.master_report import generate_master_report
from transform_ai_detection.visualization.plot_all import generate_all_plots
from transform_ai_detection.models.registry import ModelRegistry
from transform_ai_detection.transformations.base import TransformationRegistry
from transform_ai_detection.dataset.validator import DatasetValidator
from transform_ai_detection.dataset.synthetic import SyntheticDatasetAdapter
from transform_ai_detection.dataset.tampered_combined import TamperedCombinedDatasetAdapter
from transform_ai_detection.hardware import detect_hardware

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transform_ai_detection")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute an experiment from YAML configuration."""
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if getattr(args, "skip_existing_models", None) is not None:
        config.models.skip_existing_models = args.skip_existing_models
    if getattr(args, "skip_oom_models", None) is not None:
        config.models.skip_oom_models = args.skip_oom_models
    if getattr(args, "skip_models", None):
        config.models.skip_models = list(dict.fromkeys(config.models.skip_models + args.skip_models))

    runner = ExperimentRunner(config)
    summary = runner.run()

    if config.output.save_visualizations:
        logger.info("Generating publication-quality figures...")
        generate_all_plots(config.output.output_dir, export_formats=config.output.export_formats)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"Results Directory: {config.output.output_dir}")
    print(f"Total Evaluations: {summary.get('total_evaluations_completed')}")
    print(f"Overall Failure Rate: {summary.get('overall_failure_rate', 0.0):.2%}")
    print("=" * 60 + "\n")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate all figures from an existing results directory without inference."""
    output_dir = os.path.abspath(args.results_dir)
    logger.info(f"Generating publication plots from results at: {output_dir}")
    generate_all_plots(output_dir, export_formats=["png", "pdf"])
    logger.info("Plotting complete.")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate or regenerate the comprehensive master Markdown report from disk results."""
    output_dir = os.path.abspath(args.results_dir)
    logger.info(f"Generating master Markdown research report from: {output_dir}")
    report_file = generate_master_report(output_dir)
    print(f"\nMaster report generated successfully: {report_file}\n")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run an end-to-end CPU-only synthetic smoke test."""
    logger.info("Executing zero-download CPU-only synthetic smoke test...")
    cfg = ExperimentConfig(
        name="synthetic_smoke_test",
        description="CPU smoke test using synthetic tampered images and forensic baselines",
        dataset=DatasetConfig(
            adapter="synthetic",
            max_samples=10,
            seed=42,
        ),
        transformations=TransformationConfig(
            enabled_transforms=["identity", "jpeg_compression", "gaussian_blur", "downscale_upscale"],
            severities={
                "jpeg_compression": [75, 50],
                "gaussian_blur": [1.5],
                "downscale_upscale": [0.5],
            },
        ),
        models=ModelConfig(
            models=["ela_detector", "noise_residual_detector", "synthetic_oracle"],
            device="cpu",
        ),
    )
    cfg.output.output_dir = args.output_dir or "results/smoke_test"

    runner = ExperimentRunner(cfg)
    summary = runner.run()

    logger.info("Generating smoke test plots...")
    generate_all_plots(cfg.output.output_dir)

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {cfg.output.output_dir}")
    print(f"Total Evaluations: {summary.get('total_evaluations_completed')}")
    print("=" * 60 + "\n")


def cmd_validate_dataset(args: argparse.Namespace) -> None:
    """Validate a sample of rows from the Hugging Face dataset."""
    logger.info(f"Validating dataset {args.repo} (split: {args.split}, samples: {args.samples})...")
    validator = DatasetValidator()
    adapter = TamperedCombinedDatasetAdapter(
        repo_id=args.repo,
        split=args.split,
        streaming=True,
        validator=validator,
    )
    indices = adapter.deterministic_sample_indices(count=args.samples, seed=42)

    valid_count = 0
    for idx in indices:
        s = adapter[idx]
        if s.metadata.get("is_valid", True):
            valid_count += 1

    summary = validator.exclusion_logger.summary()
    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print(f"Total Inspected: {len(indices)}")
    print(f"Valid Samples: {valid_count}")
    print(f"Excluded Samples: {len(indices) - valid_count}")
    print(f"Exclusion Reasons Breakdown: {summary}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iml-robustness",
        description="Transformation Robustness and Failure Analysis for AI Image Manipulation Localization",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    p_run = subparsers.add_parser("run", help="Run an experiment from YAML config")
    p_run.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    p_run.add_argument("--output-dir", "-o", default=None, help="Override output directory")
    p_run.add_argument("--max-samples", "-n", type=int, default=None, help="Override sample count")
    p_run.add_argument(
        "--skip-existing-models",
        dest="skip_existing_models",
        action="store_true",
        default=True,
        help="Detect and skip models that already have completed evaluations in results directory (default: True)",
    )
    p_run.add_argument(
        "--no-skip-existing-models",
        dest="skip_existing_models",
        action="store_false",
        help="Force re-evaluation of models even if existing evaluations are present in results directory",
    )
    p_run.add_argument(
        "--skip-oom-models",
        dest="skip_oom_models",
        action="store_true",
        default=True,
        help="Detect and skip models that encounter CUDA/system OutOfMemory errors (default: True)",
    )
    p_run.add_argument(
        "--no-skip-oom-models",
        dest="skip_oom_models",
        action="store_false",
        help="Disable skipping of models that encounter OutOfMemory errors (will retry on subsequent samples)",
    )
    p_run.add_argument(
        "--skip-models",
        nargs="+",
        default=None,
        help="Explicit list of model names to skip from evaluation (e.g. --skip-models trufor sparsevit)",
    )
    p_run.set_defaults(func=cmd_run)

    # plot command
    p_plot = subparsers.add_parser("plot", help="Generate publication plots from saved results")
    p_plot.add_argument("--results-dir", "-r", required=True, help="Path to results directory")
    p_plot.set_defaults(func=cmd_plot)

    # report command
    p_rep = subparsers.add_parser("report", help="Generate master Markdown report from saved results")
    p_rep.add_argument("--results-dir", "-r", required=True, help="Path to results directory")
    p_rep.set_defaults(func=cmd_report)

    # smoke-test command
    p_smoke = subparsers.add_parser("smoke-test", help="Run quick synthetic CPU smoke test")
    p_smoke.add_argument("--output-dir", "-o", default="results/smoke_test", help="Output directory")
    p_smoke.set_defaults(func=cmd_smoke_test)

    # validate-dataset command
    p_val = subparsers.add_parser("validate-dataset", help="Validate remote dataset schema and samples")
    p_val.add_argument("--repo", default="KhangTruong/Tampered-Images-Combined-Dataset")
    p_val.add_argument("--split", default="cross_test")
    p_val.add_argument("--samples", type=int, default=20)
    p_val.set_defaults(func=cmd_validate_dataset)

    # hardware command
    p_hw = subparsers.add_parser("hardware", help="Print detected hardware and runtime specs")
    p_hw.set_defaults(func=lambda a: print(json.dumps(detect_hardware(), indent=2)))

    # list-models command
    p_models = subparsers.add_parser("list-models", help="List registered models")
    p_models.set_defaults(func=lambda a: print("Registered models:\n  - " + "\n  - ".join(ModelRegistry.list_models())))

    # list-transforms command
    p_t = subparsers.add_parser("list-transforms", help="List registered transformations")
    p_t.set_defaults(func=lambda a: print("Registered transformations:\n  - " + "\n  - ".join(TransformationRegistry.list_available())))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
