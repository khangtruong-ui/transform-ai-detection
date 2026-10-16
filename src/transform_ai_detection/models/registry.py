"""Model registry and availability reporting."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from transform_ai_detection.models.base import BaseLocalizationModel
from transform_ai_detection.models.baselines import (
    ErrorLevelAnalysisModel,
    NoiseResidualModel,
    DCTInconsistencyModel,
)
from transform_ai_detection.models.deep_forensic import DeepForensicModel
from transform_ai_detection.models.synthetic import (
    SyntheticOracleModel,
    MockConstantModel,
)
from transform_ai_detection.models.trufor import TruForAdapter
from transform_ai_detection.models.sparse_vit import SparseViTModel
from transform_ai_detection.models.iml_vit import IMLViTModel
from transform_ai_detection.models.pscc_net import PSCCNetModel
from transform_ai_detection.models.code_net import CoDEModel
from transform_ai_detection.models.mask_clip import MaskCLIPModel
from transform_ai_detection.models.sida import SIDAModel
from transform_ai_detection.models.rita import RITAModel
from transform_ai_detection.hardware import get_torch_device


class ModelRegistry:
    """Registry for instantiating and inspecting localization models."""

    _factories: Dict[str, Callable[..., BaseLocalizationModel]] = {
        # Baselines
        "ela_detector": lambda **kw: ErrorLevelAnalysisModel(**kw),
        "noise_residual_detector": lambda **kw: NoiseResidualModel(**kw),
        "dct_inconsistency_detector": lambda **kw: DCTInconsistencyModel(**kw),
        "deep_forensic_detector": lambda **kw: DeepForensicModel(**kw),
        "synthetic_oracle": lambda **kw: SyntheticOracleModel(name="synthetic_oracle", **kw),
        "synthetic_noisy": lambda **kw: SyntheticOracleModel(name="synthetic_noisy", noise_level=0.2, **kw),
        "mock_zeros": lambda **kw: MockConstantModel(constant_value=0, name="mock_zeros"),
        "mock_ones": lambda **kw: MockConstantModel(constant_value=1, name="mock_ones"),

        # Integrated Research Models (arxiv:2608.20929)
        "sparsevit": lambda **kw: SparseViTModel(**kw),
        "sparse_vit": lambda **kw: SparseViTModel(**kw),
        "iml_vit": lambda **kw: IMLViTModel(**kw),
        "iml-vit": lambda **kw: IMLViTModel(**kw),
        "pscc_net": lambda **kw: PSCCNetModel(**kw),
        "pscc-net": lambda **kw: PSCCNetModel(**kw),
        "pscc": lambda **kw: PSCCNetModel(**kw),
        "trufor": lambda **kw: TruForAdapter(**kw),
        "code": lambda **kw: CoDEModel(**kw),
        "maskclip": lambda **kw: MaskCLIPModel(**kw),
        "mask_clip": lambda **kw: MaskCLIPModel(**kw),
        "sida": lambda **kw: SIDAModel(**kw),
        "rita": lambda **kw: RITAModel(**kw),
    }

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseLocalizationModel]) -> None:
        cls._factories[name] = factory

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._factories

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseLocalizationModel:
        if name not in cls._factories:
            raise KeyError(f"Model '{name}' not found. Available: {list(cls._factories.keys())}")
        return cls._factories[name](**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(list(cls._factories.keys()))

    @classmethod
    def list_integrated_paper_models(cls) -> List[str]:
        return ["sparsevit", "iml_vit", "pscc_net", "trufor", "code", "maskclip", "sida", "rita"]

    @classmethod
    def list_classical_models(cls) -> List[str]:
        return [
            "ela_detector",
            "noise_residual_detector",
            "dct_inconsistency_detector",
            "deep_forensic_detector",
        ]

    @classmethod
    def get_model_category(cls, name: str) -> str:
        """Classify a model name into 'Classical Models' or 'Integrated Models'."""
        clean_name = name.lower().replace("-", "_")
        integrated = {m.lower().replace("-", "_") for m in cls.list_integrated_paper_models()}
        integrated.update({"sparse_vit", "iml_vit", "pscc_net", "pscc", "mask_clip"})
        if clean_name in integrated:
            return "Integrated Models"
        return "Classical Models"

    @classmethod
    def is_integrated_model(cls, name: str) -> bool:
        return cls.get_model_category(name) == "Integrated Models"

    @classmethod
    def is_classical_model(cls, name: str) -> bool:
        return cls.get_model_category(name) == "Classical Models"

    @classmethod
    def generate_availability_report(
        cls,
        model_names: List[str],
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_device: str = "auto",
    ) -> List[Dict[str, Any]]:
        model_kwargs = model_kwargs or {}
        report = []
        for name in model_names:
            kw = dict(model_kwargs.get(name, {}))
            if "device" not in kw:
                kw["device"] = default_device
            try:
                model = cls.create(name, **kw)
                info = model.get_model_info()
                info["category"] = cls.get_model_category(name)
                info["registered"] = True
                info["load_successful"] = model.is_available()
                report.append(info)
            except Exception as e:
                report.append({
                    "name": name,
                    "category": cls.get_model_category(name),
                    "registered": name in cls._factories,
                    "load_successful": False,
                    "error": str(e),
                })
        return report
