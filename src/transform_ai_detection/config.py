"""Configuration classes and YAML/JSON loader for experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DatasetConfig:
    name: str = "KhangTruong/Tampered-Images-Combined-Dataset"
    adapter: str = "tampered_combined"  # "tampered_combined", "synthetic", "local"
    split: str = "cross_test"
    max_samples: Optional[int] = 100
    sample_indices: Optional[List[int]] = None
    seed: int = 42
    streaming: bool = True
    max_cached_row_groups: int = 2
    filter_methods: Optional[List[str]] = None
    filter_datasets: Optional[List[str]] = None
    cache_dir: Optional[str] = None
    local_data_dir: Optional[str] = None


@dataclass
class TransformationConfig:
    enabled_transforms: List[str] = field(
        default_factory=lambda: [
            "identity",
            "jpeg_compression",
            "gaussian_blur",
            "downscale_upscale",
            "brightness_contrast",
        ]
    )
    severities: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "jpeg_compression": [95, 75, 50, 25],
            "gaussian_blur": [1.0, 2.0, 3.0, 5.0],
            "downscale_upscale": [0.75, 0.5, 0.25],
            "brightness_contrast": [
                {"factor_brightness": 1.2, "factor_contrast": 1.2},
                {"factor_brightness": 0.8, "factor_contrast": 0.8},
                {"factor_brightness": 1.5, "factor_contrast": 1.5},
            ],
            "gaussian_noise": [0.01, 0.03, 0.05],
            "sharpening": [1.5, 2.0, 3.0],
            "horizontal_flip": [True],
        }
    )
    seed: int = 42
    include_geometric: bool = False
    crop_size: Optional[List[int]] = None


@dataclass
class ModelConfig:
    models: List[str] = field(
        default_factory=lambda: [
            "ela_detector",
            "noise_residual_detector",
            "dct_inconsistency_detector",
        ]
    )
    device: str = "auto"
    batch_size: int = 1
    inference_resolution: Optional[List[int]] = None  # None: keep native image resolution
    model_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skip_existing_models: bool = True
    skip_oom_models: bool = True
    skip_models: List[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    threshold_policy: str = "fixed_0.5"
    fixed_threshold: float = 0.5
    boundary_tolerance: int = 3
    compute_ranking_metrics: bool = True
    bootstrap_iterations: int = 1000
    bootstrap_ci: float = 0.95
    failure_metric: str = "iou"
    failure_threshold: float = 0.3
    degradation_failure_delta: float = -0.2
    run_semantic_boundary_diagnostic: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "results/default_experiment"
    save_predictions: bool = False
    save_visualizations: bool = True
    save_per_image_diagnostics: bool = True
    plot_dpi: int = 300
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class ExperimentConfig:
    name: str = "iml_robustness_study"
    description: str = "Study of classical transformations on IML models"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transformations: TransformationConfig = field(default_factory=TransformationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        data = dict(data)
        d_cfg = DatasetConfig(**data.get("dataset", {}))
        t_cfg = TransformationConfig(**data.get("transformations", {}))
        m_cfg = ModelConfig(**data.get("models", {}))
        e_cfg = EvaluationConfig(**data.get("evaluation", {}))
        o_cfg = OutputConfig(**data.get("output", {}))
        return cls(
            name=data.get("name", "iml_robustness_study"),
            description=data.get("description", "IML Robustness Study"),
            dataset=d_cfg,
            transformations=t_cfg,
            models=m_cfg,
            evaluation=e_cfg,
            output=o_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ExperimentConfig:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
