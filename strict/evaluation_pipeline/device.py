from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit accelerator, including ROCm's CUDA-compatible API."""
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    elif name == "rocm":
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("ROCm requested, but this PyTorch build has no HIP device")
        name = "cuda"

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"accelerator {requested!r} requested, but CUDA/ROCm is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested, but it is unavailable")
    return device


def runtime_info(device: torch.device) -> dict[str, object]:
    backend = "rocm" if device.type == "cuda" and torch.version.hip else device.type
    value: dict[str, object] = {
        "schema_version": 1,
        "backend": backend,
        "device": str(device),
        "device_name": None,
        "device_count": 0,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        value.update(
            device_name=torch.cuda.get_device_name(index),
            device_count=torch.cuda.device_count(),
        )
    elif device.type == "mps":
        value.update(device_name="Apple MPS", device_count=1)
    return value


def write_runtime_info(path: Path, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_info(device), indent=2, sort_keys=True) + "\n")
