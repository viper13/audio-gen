"""Pre-converted mlx-Yue2-3B weight inspection and background Hugging Face downloader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import threading
import time
from typing import Callable

from .config import (
    DEFAULT_MODEL_DIR,
    DEFAULT_VAE_DIR,
    HF_PRECONVERTED_REPO,
    REQUIRED_MODEL_FILES,
    REQUIRED_VAE_FILES,
)


def normalize_downloaded_layout(model_dir: Path = DEFAULT_MODEL_DIR, vae_dir: Path = DEFAULT_VAE_DIR) -> None:
    """Keep models/mlx-Yue2-3B strictly matching conversion.json (move vae/ out and drop HF metadata)."""
    nested_vae = model_dir / "vae"
    if nested_vae.is_dir() and nested_vae.resolve() != vae_dir.resolve():
        if vae_dir.exists():
            shutil.rmtree(vae_dir)
        vae_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(nested_vae), str(vae_dir))

    for extra_dir in (".cache",):
        p = model_dir / extra_dir
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    for extra_file in (".gitattributes",):
        p = model_dir / extra_file
        if p.exists():
            p.unlink(missing_ok=True)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def inspect_models(model_dir: Path = DEFAULT_MODEL_DIR, vae_dir: Path = DEFAULT_VAE_DIR) -> dict:
    """Check whether local MLX YuE2-3B weights are complete and valid."""
    missing = [rel for rel in REQUIRED_MODEL_FILES if not (model_dir / rel).is_file()]
    missing.extend(f"vae/{rel}" for rel in REQUIRED_VAE_FILES if not (vae_dir / rel).is_file())
    available_precisions = [
        p for p in ("8bit", "bf16", "4bit") if (model_dir / f"ar-{p}.safetensors").is_file()
    ]
    ready = len(missing) == 0
    size_bytes = _dir_size_bytes(model_dir)
    if vae_dir.exists() and not vae_dir.resolve().is_relative_to(model_dir.resolve()):
        size_bytes += _dir_size_bytes(vae_dir)

    return {
        "ready": ready,
        "repo_id": HF_PRECONVERTED_REPO,
        "model_dir": str(model_dir),
        "vae_dir": str(vae_dir),
        "available_precisions": available_precisions,
        "missing_files": missing,
        "disk_size_gib": round(size_bytes / (2**30), 2),
        "expected_size_gib": 10.6,
    }


@dataclass
class DownloadState:
    status: str = "idle"  # idle | downloading | completed | failed
    message: str = ""
    downloaded_gib: float = 0.0
    expected_gib: float = 10.6
    percent: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    files_done: int = 0
    files_total: int = len(REQUIRED_MODEL_FILES) + len(REQUIRED_VAE_FILES)

    def to_dict(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at or time.time()
            elapsed = round(max(0.0, end - self.started_at), 1)
        return {
            "status": self.status,
            "message": self.message,
            "downloaded_gib": round(self.downloaded_gib, 2),
            "expected_gib": self.expected_gib,
            "percent": round(min(100.0, max(0.0, self.percent)), 1),
            "elapsed_seconds": elapsed,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "error": self.error,
        }


class ModelDownloadManager:
    """Runs a single background snapshot_download of vanch007/mlx-Yue2-3B with live disk polling."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        vae_dir: Path = DEFAULT_VAE_DIR,
        on_change: Callable[[dict], None] | None = None,
    ):
        self.model_dir = model_dir
        self.vae_dir = vae_dir
        self._on_change = on_change
        self._lock = threading.Lock()
        self._state = DownloadState()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self._lock:
            state = self._state.to_dict()
        return {**inspect_models(self.model_dir, self.vae_dir), "download": state}

    def _emit(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change(self.snapshot())
            except Exception:
                pass

    def start_download(self) -> dict:
        total_files = len(REQUIRED_MODEL_FILES) + len(REQUIRED_VAE_FILES)
        with self._lock:
            if self._state.status == "downloading":
                return self.snapshot()
            info = inspect_models(self.model_dir, self.vae_dir)
            if info["ready"]:
                self._state = DownloadState(
                    status="completed",
                    message="Model weights are already downloaded and ready.",
                    downloaded_gib=info["disk_size_gib"],
                    percent=100.0,
                    files_done=total_files,
                )
                return self.snapshot()

            self._state = DownloadState(
                status="downloading",
                message=f"Downloading {HF_PRECONVERTED_REPO} (~10.6 GB) from Hugging Face...",
                started_at=time.time(),
            )
            self._thread = threading.Thread(target=self._worker, name="yue2-model-download", daemon=True)
            self._thread.start()
        self._emit()
        return self.snapshot()

    def _worker(self) -> None:
        stop_poll = threading.Event()
        total_files = len(REQUIRED_MODEL_FILES) + len(REQUIRED_VAE_FILES)

        def poll_disk() -> None:
            while not stop_poll.wait(1.0):
                size_gib = _dir_size_bytes(self.model_dir) / (2**30)
                done = sum(1 for rel in REQUIRED_MODEL_FILES if (self.model_dir / rel).is_file())
                done += sum(
                    1
                    for rel in REQUIRED_VAE_FILES
                    if (self.model_dir / "vae" / rel).is_file() or (self.vae_dir / rel).is_file()
                )
                with self._lock:
                    if self._state.status != "downloading":
                        return
                    self._state.downloaded_gib = size_gib
                    self._state.files_done = done
                    self._state.percent = min(99.0, (size_gib / self._state.expected_gib) * 100.0)
                    self._state.message = (
                        f"Downloading {HF_PRECONVERTED_REPO}... "
                        f"{size_gib:.2f} / ~{self._state.expected_gib:.1f} GB ({done}/{total_files} files)"
                    )
                self._emit()

        poller = threading.Thread(target=poll_disk, name="yue2-download-poll", daemon=True)
        poller.start()
        try:
            from huggingface_hub import snapshot_download

            self.model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=HF_PRECONVERTED_REPO,
                local_dir=str(self.model_dir),
            )
            normalize_downloaded_layout(self.model_dir, self.vae_dir)
            info = inspect_models(self.model_dir, self.vae_dir)
            with self._lock:
                self._state.finished_at = time.time()
                self._state.downloaded_gib = info["disk_size_gib"]
                self._state.files_done = total_files - len(info["missing_files"])
                if info["ready"]:
                    self._state.status = "completed"
                    self._state.percent = 100.0
                    self._state.message = "YuE2-3B MLX weights downloaded and verified!"
                    self._state.error = None
                else:
                    self._state.status = "failed"
                    self._state.error = f"Download finished but missing files: {info['missing_files']}"
                    self._state.message = self._state.error
        except Exception as exc:
            with self._lock:
                self._state.status = "failed"
                self._state.finished_at = time.time()
                self._state.error = str(exc)
                self._state.message = f"Download failed: {exc}"
        finally:
            stop_poll.set()
            poller.join(timeout=2.0)
            self._emit()
