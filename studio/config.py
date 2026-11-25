"""Paths, hardware detection, and MLX / PyTorch CUDA defaults for YuE2 Studio."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import sys
import psutil

# Must be set before mlx.core is ever imported.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT_DIR / "engine"
ENGINE_SRC = ENGINE_DIR / "src"
VENDOR_YUE_SRC = ENGINE_DIR / "vendor" / "yue" / "src"
for _src_dir in (VENDOR_YUE_SRC, ENGINE_SRC):
    if _src_dir.is_dir() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

MODELS_DIR = Path(os.environ.get("STUDIO_MODELS_DIR") or (ROOT_DIR / "models")).resolve()
OUTPUTS_DIR = Path(os.environ.get("STUDIO_OUTPUTS_DIR") or (ROOT_DIR / "outputs")).resolve()
CACHE_DIR = Path(os.environ.get("STUDIO_CACHE_DIR") or (ROOT_DIR / ".cache")).resolve()
HF_CACHE_DIR = Path(os.environ.get("HF_HOME") or (MODELS_DIR / "hf-cache")).resolve()
TMP_DIR = (CACHE_DIR / "tmp").resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Keep all Hugging Face, PyTorch, and temp caches on the workspace drive (e.g. D:) so C: is never filled.
for _dir in (MODELS_DIR, CACHE_DIR, HF_CACHE_DIR, HF_CACHE_DIR / "hub", CACHE_DIR / "torch", CACHE_DIR / "xdg", TMP_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
if not os.environ.get("STUDIO_KEEP_SYSTEM_TEMP"):
    import tempfile

    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)
    tempfile.tempdir = str(TMP_DIR)


def detect_backend() -> str:
    """Return 'mlx' on Apple Silicon macOS or 'torch' on Windows/Linux (override via STUDIO_BACKEND)."""
    override = os.environ.get("STUDIO_BACKEND", "").strip().lower()
    if override in {"mlx", "torch"}:
        return override
    if platform.system() == "Darwin" and platform.machine().lower() == "arm64":
        return "mlx"
    return "torch"


BACKEND = detect_backend()

# Hugging Face repositories
HF_PRECONVERTED_REPO = "vanch007/mlx-Yue2-3B"
HF_TORCH_MODEL_REPO = "m-a-p/YuE2-3B"
HF_TORCH_VAE_REPO = "m-a-p/YuE2-Vae"

if BACKEND == "mlx":
    DEFAULT_MODEL_DIR = MODELS_DIR / "mlx-Yue2-3B"
    DEFAULT_VAE_DIR = MODELS_DIR / "vae"
else:
    DEFAULT_MODEL_DIR = MODELS_DIR / "YuE2-3B"
    DEFAULT_VAE_DIR = MODELS_DIR / "YuE2-Vae"

# Required files inside models/mlx-Yue2-3B for 8-bit & BF16 MLX inference
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
    """Compute a safe GPU memory budget (Apple Silicon unified memory or NVIDIA CUDA VRAM)."""
    if BACKEND == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (2**30)
                return round(min(24.0, max(4.0, total_vram_gib)), 2)
        except Exception:
            pass
        total_gib = psutil.virtual_memory().total / (2**30)
        return round(min(24.0, max(6.0, total_gib - 4.0)), 2)

    total_gib = psutil.virtual_memory().total / (2**30)
    # GPUExecution requires: 5 < budget <= total_gib - 4
    max_allowed = max(5.5, total_gib - 4.0)
    # Cap at 16 GiB (mlx-Yue's default tested ceiling)
    return round(min(16.0, max_allowed), 2)


def get_system_status() -> dict:
    """Return live memory and GPU (Apple Silicon MLX or NVIDIA CUDA) readiness."""
    vm = psutil.virtual_memory()
    try:
        swap_used_gib = round(psutil.swap_memory().used / (2**30), 2)
    except Exception:
        swap_used_gib = 0.0
    total_gib = vm.total / (2**30)
    available_gib = vm.available / (2**30)
    used_gib = (vm.total - vm.available) / (2**30)

    runtime_info: dict = {
        "backend": BACKEND,
        "supported": False,
        "metal": False,
        "cuda": False,
        "device_name": platform.machine(),
        "macos": platform.mac_ver()[0],
        "error": None,
    }
    vram_total_gib: float | None = None
    vram_available_gib: float | None = None

    if BACKEND == "mlx":
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
    else:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                vram_total_gib = round(total_bytes / (2**30), 2)
                vram_available_gib = round(free_bytes / (2**30), 2)
                bf16_ok = bool(torch.cuda.is_bf16_supported())
                runtime_info.update(
                    supported=bf16_ok,
                    cuda=True,
                    device_name=props.name,
                    architecture=f"CUDA sm_{props.major}{props.minor}",
                    compute_capability=[props.major, props.minor],
                    torch_version=torch.__version__,
                    error=None if bf16_ok else "CUDA GPU does not support BF16",
                )
            else:
                runtime_info.update(
                    supported=True,
                    cuda=False,
                    device_name=f"CPU ({platform.processor() or platform.machine()})",
                    torch_version=torch.__version__,
                    error="No CUDA GPU detected (will fall back to slow CPU execution)",
                )
        except Exception as exc:
            runtime_info["error"] = str(exc)
        low_memory_warning = (
            (vram_available_gib is not None and vram_available_gib < 5.0)
            or available_gib < 4.0
        )

    memory_payload: dict = {
        "total_gib": round(total_gib, 2),
        "available_gib": round(available_gib, 2),
        "used_gib": round(used_gib, 2),
        "percent_used": round(vm.percent, 1),
        "swap_used_gib": swap_used_gib,
        "recommended_budget_gib": compute_safe_memory_budget_gib(),
        "low_memory_warning": low_memory_warning,
    }
    if vram_total_gib is not None:
        memory_payload["vram_total_gib"] = vram_total_gib
        memory_payload["vram_available_gib"] = vram_available_gib

    return {
        "platform": platform.platform(),
        "backend": BACKEND,
        "memory": memory_payload,
        "runtime": runtime_info,
    }
