"""Paths, hardware detection, and Apple Silicon safe defaults for YuE2 Studio."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import psutil

# Must be set before mlx.core is ever imported.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT_DIR / "engine"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "mlx-Yue2-3B"
DEFAULT_VAE_DIR = MODELS_DIR / "vae"
OUTPUTS_DIR = ROOT_DIR / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

HF_PRECONVERTED_REPO = "vanch007/mlx-Yue2-3B"

# Required files inside models/mlx-Yue2-3B for 8-bit & BF16 inference
REQUIRED_MODEL_FILES = (
    "conversion.json",
    "config.json",
    "qwen.tiktoken",
    "ar-8bit.safetensors",
    "ar-bf16.safetensors",
    "nar-bf16.safetensors",
)

# Required files inside models/vae (kept separate from mlx-Yue2-3B so lyra's strict manifest check passes)
REQUIRED_VAE_FILES = (
    "config.json",
    "model.safetensors",
)


def compute_safe_memory_budget_gib() -> float:
    """Compute the largest valid GPUExecution budget leaving >= 4 GiB OS headroom."""
    total_gib = psutil.virtual_memory().total / (2**30)
    # GPUExecution requires: 5 < budget <= total_gib - 4
    max_allowed = max(5.5, total_gib - 4.0)
    # Cap at 16 GiB (mlx-Yue's default tested ceiling)
    return round(min(16.0, max_allowed), 2)


def get_system_status() -> dict:
    """Return live Apple Silicon memory and MLX Metal readiness."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total_gib = vm.total / (2**30)
    available_gib = vm.available / (2**30)
    used_gib = (vm.total - vm.available) / (2**30)

    runtime_info = {
        "supported": False,
        "metal": False,
        "device_name": platform.machine(),
        "macos": platform.mac_ver()[0],
        "error": None,
    }
    try:
        from lyra.runtime import runtime_status

        rt = runtime_status()
        device = rt.get("device") or {}
        runtime_info.update(
            supported=bool(rt.get("supported")),
            metal=bool(rt.get("metal")),
            device_name=device.get("device_name") or platform.machine(),
            architecture=device.get("architecture", ""),
            macos=rt.get("macos", platform.mac_ver()[0]),
            error=rt.get("error"),
        )
    except Exception as exc:
        runtime_info["error"] = str(exc)

    low_memory_warning = available_gib < 4.5
    return {
        "platform": platform.platform(),
        "memory": {
            "total_gib": round(total_gib, 2),
            "available_gib": round(available_gib, 2),
            "used_gib": round(used_gib, 2),
            "percent_used": round(vm.percent, 1),
            "swap_used_gib": round(swap.used / (2**30), 2),
            "recommended_budget_gib": compute_safe_memory_budget_gib(),
            "low_memory_warning": low_memory_warning,
        },
        "runtime": runtime_info,
    }
