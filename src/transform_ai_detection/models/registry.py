"""Model registry and availability reporting."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type
from transform_ai_detection.models.base import BaseLocalizationModel
from transform_ai_detection.models.baselines import (
    ErrorLevelAnalysisModel,
    NoiseResidualModel,
    DCTInconsistencyModel,
)
from transform_ai_detection.models.synthetic import (
    SyntheticOracleModel,
    MockConstantModel,
)
from transform_ai_detection.models.trufor import TruForAdapter


class ModelRegistry:
    """Registry for instantiating and inspecting localization models."""

    _factories: Dict[str, Callable[..., BaseLocalizationModel]] = {
        "ela_detector": lambda **kw: ErrorLevelAnalysisModel(**kw),
        "noise_residual_detector": lambda **kw: NoiseResidualModel(**kw),
        "dct_inconsistency_detector": lambda **kw: DCTInconsistencyModel(**kw),
        "trufor": lambda **kw: TruForAdapter(**kw),
        "synthetic_oracle": lambda **kw: SyntheticOracleModel(name="synthetic_oracle", **kw),
        "synthetic_noisy": lambda **kw: SyntheticOracleModel(name="synthetic_noisy", noise_level=0.2, **kw),
        "mock_zeros": lambda **kw: MockConstantModel(constant_value=0, name="mock_zeros"),
        "mock_ones": lambda **kw: MockConstantModel(constant_value=1, name="mock_ones"),
    }

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseLocalizationModel]) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseLocalizationModel:
        if name not in cls._factories:
            raise KeyError(f"Model '{name}' not found. Available: {list(cls._factories.keys())}")
        return cls._factories[name](**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(list(cls._factories.keys()))

    @classmethod
    def generate_availability_report(
        cls, model_names: List[str], model_kwargs: Dict[str, Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        model_kwargs = model_kwargs or {}
        report = []
        for name in model_names:
            kw = model_kwargs.get(name, {})
            try:
                model = cls.create(name, **kw)
                info = model.get_model_info()
                info["registered"] = True
                info["load_successful"] = model.is_available()
                report.append(info)
            except Exception as e:
                report.append({
                    "name": name,
                    "registered": name in cls._factories,
                    "load_successful": False,
                    "error": str(e),
                })
        return report
