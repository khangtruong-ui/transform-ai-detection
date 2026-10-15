"""Base transformation interfaces, metadata models, and registry."""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Type
import numpy as np
from PIL import Image


@dataclass
class TransformMetadata:
    """Explicit declaration of transformation characteristics and interpretation bounds."""

    name: str
    severity_level: Any
    severity_label: str
    changes_geometry: bool
    changes_dimensions: bool
    requires_mask_interpolation: bool
    mask_interpolation_mode: str  # "none", "nearest", etc.
    parameters: Dict[str, Any]
    limitations: str

    @property
    def deterministic_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "severity_level": str(self.severity_level),
                "parameters": self.parameters,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["deterministic_hash"] = self.deterministic_hash
        return d


@dataclass
class TransformResult:
    """Output of applying a transformation to an image and its ground truth mask."""

    image: Image.Image
    mask: np.ndarray  # (H, W) uint8 {0, 1}
    metadata: TransformMetadata
    is_valid_comparison: bool = True
    invalidation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mask.ndim == 3:
            self.mask = self.mask[:, :, 0]
        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.uint8)
        else:
            self.mask = (self.mask > 0).astype(np.uint8)


class BaseTransformation(abc.ABC):
    """Abstract base class for all transformations."""

    @abc.abstractmethod
    def apply(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        sample_id: str = "",
    ) -> TransformResult:
        """Apply transformation deterministically to image and mask."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_metadata(self) -> TransformMetadata:
        """Retrieve transformation metadata and parameter specs."""
        raise NotImplementedError


class TransformationRegistry:
    """Registry for registering and instantiating transformation instances."""

    _registry: Dict[str, Type[BaseTransformation]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        def decorator(subclass: Type[BaseTransformation]) -> Type[BaseTransformation]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTransformation]:
        if name not in cls._registry:
            raise KeyError(
                f"Transformation '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))
