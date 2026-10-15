"""Transformation Robustness and Failure Analysis for AI Image Manipulation Localization."""

__version__ = "0.1.0"

from transform_ai_detection.config import (
    ExperimentConfig,
    DatasetConfig,
    TransformationConfig,
    ModelConfig,
    EvaluationConfig,
    OutputConfig,
)
from transform_ai_detection.hardware import detect_hardware, get_torch_device
from transform_ai_detection.dataset import (
    BaseDatasetAdapter,
    DatasetSample,
    DatasetValidator,
    SyntheticDatasetAdapter,
    TamperedCombinedDatasetAdapter,
)
from transform_ai_detection.transformations import (
    BaseTransformation,
    TransformMetadata,
    TransformResult,
    TransformationRegistry,
)
from transform_ai_detection.models import (
    BaseLocalizationModel,
    ModelOutput,
    ModelRegistry,
)
from transform_ai_detection.metrics import (
    PixelMetrics,
    BoundaryMetrics,
    EvaluationEngine,
)
from transform_ai_detection.diagnostics import (
    GeometricDiagnostics,
    TextureDiagnostics,
    PostInferenceDiagnostics,
)
from transform_ai_detection.failure_analysis import (
    FailureClassification,
    FailureClassifier,
    analyze_shared_failures,
    stratify_performance,
)
from transform_ai_detection.experiment import (
    ExperimentRunner,
    ExperimentLedger,
    ExperimentSummaryGenerator,
)
from transform_ai_detection.visualization import generate_all_plots

__all__ = [
    "ExperimentConfig",
    "DatasetConfig",
    "TransformationConfig",
    "ModelConfig",
    "EvaluationConfig",
    "OutputConfig",
    "detect_hardware",
    "get_torch_device",
    "BaseDatasetAdapter",
    "DatasetSample",
    "DatasetValidator",
    "SyntheticDatasetAdapter",
    "TamperedCombinedDatasetAdapter",
    "BaseTransformation",
    "TransformMetadata",
    "TransformResult",
    "TransformationRegistry",
    "BaseLocalizationModel",
    "ModelOutput",
    "ModelRegistry",
    "PixelMetrics",
    "BoundaryMetrics",
    "EvaluationEngine",
    "GeometricDiagnostics",
    "TextureDiagnostics",
    "PostInferenceDiagnostics",
    "FailureClassification",
    "FailureClassifier",
    "analyze_shared_failures",
    "stratify_performance",
    "ExperimentRunner",
    "ExperimentLedger",
    "ExperimentSummaryGenerator",
    "generate_all_plots",
]
