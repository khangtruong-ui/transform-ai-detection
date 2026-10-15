"""Hardware detection and environment inspection utilities."""

from __future__ import annotations

import os
import platform
import psutil
from typing import Any, Dict
import torch


def detect_hardware() -> Dict[str, Any]:
    """Inspect and return complete hardware and runtime environment specs."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count() or 1,
            "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_bytes": psutil.virtual_memory().available,
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "disk": {
            "total_gb": round(psutil.disk_usage(".").total / (1024**3), 2),
            "free_gb": round(psutil.disk_usage(".").free / (1024**3), 2),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })
        info["pytorch"]["gpus"] = gpu_info
    else:
        info["pytorch"]["gpus"] = []

    return info


def get_torch_device(requested: str = "auto") -> torch.device:
    """Resolve torch device based on preference and hardware availability.

    Parameters
    ----------
    requested : str
        'auto', 'cpu', or 'cuda' (or 'cuda:0', etc.)

    Returns
    -------
    torch.device
    """
    requested = requested.lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if requested.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    
    # Auto: default to CPU if inference-only or if GPU is not available
    # Prompt states: Keep dataset access, transformations, metrics, and visualization CPU-capable.
    # Default to cuda if available and healthy, else cpu.
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
