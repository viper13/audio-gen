"""Thread-safe serial YuE2 MLX generation worker, stage progress tracker, and song library."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import re
import shutil
import threading
import time
from typing import Any, Callable
import uuid

os.environ.setdefault("MLX_ENABLE_TF32", "0")

from .config import (
    BACKEND,
    DEFAULT_MODEL_DIR,
    DEFAULT_VAE_DIR,
    OUTPUTS_DIR,
    compute_safe_memory_budget_gib,
)
from .model_manager import inspect_models


def _slugify(text: str, fallback: str = "song") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return (cleaned[:40] or fallback).strip("-") or fallback


def _format_style_instrumental(style: str) -> str:
    s = (style or "Instrumental music").strip()
    if "instrumental" not in s.lower():
        s = "Instrumental, " + s.lstrip("., ")
    for cue in ("no vocals", "no singing"):
        if cue not in s.lower():
            s = s.rstrip("., ") + f", {cue}"
    return s.rstrip("., ") + "."


@dataclass
class GenerateParams:
    title: str
    style: str
    lyrics: str
    cot: str = "melody"  # full | melody | off
    precision: str = "8bit"  # 8bit | bf16 | 4bit
    ode_steps: int = 8  # 8 (fast) | 16 | 32 (studio)
    max_tokens: int = 1200  # semantic tokens (25 tok/s => 1200 ~ 48s, 3000 ~ 120s, 6000 ~ 240s)
    min_tokens: int = 200
    seed: int | None = None
    instrumental: bool = False
    abc: str | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    cfg_scale: float | None = None
    memory_budget_gib: float | None = None


class _ProgressStageProxy:
    """Wraps yue2.progress._Stage and forwards structured progress updates to the studio UI."""

    def __init__(self, inner, label: str, unit: str | None, total: int | None, emit: Callable[[dict], None]):
        self._inner = inner
        self._label = label
        self._unit = unit
        self._total = total
        self._completed = 0
        self._started = time.monotonic()
        self._last_emit = 0.0
        self._emit = emit
        self._push(force=True)

    def _push(self, *, force: bool = False, status: str | None = None) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit) < 0.2:
            return
        self._last_emit = now
        elapsed = max(0.0, now - self._started)
        rate = self._completed / elapsed if elapsed > 0 and self._completed > 0 else 0.0
        self._emit(
            {
                "label": self._label,
                "unit": self._unit,
                "completed": self._completed,
                "total": self._total,
                "rate": round(rate, 1),
                "stage_elapsed": round(elapsed, 1),
                "stage_status": status or "running",
            }
        )

    def update(self, completed: int, total: int | None = None) -> None:
        self._inner.update(completed, total=total)
        self._completed = completed
        if total is not None:
            self._total = total
        self._push(force=(total is not None and completed == total))

    def set_total(self, total: int | None) -> None:
        self._inner.set_total(total)
        self._total = total
        self._push(force=True)

    def advance(self, count: int = 1) -> None:
        self._inner.advance(count)
        self._completed += count
        self._push(force=False)

    def token(self, phase, token) -> None:
        self._inner.token(phase, token)
        self._completed += 1
        self._push(force=False)

    def finish(self, status: str = "completed") -> None:
        self._inner.finish(status=status)
        self._push(force=True, status=status)


def _stage_key_for_label(label: str) -> str:
    low = label.lower()
    if "verif" in low or "resolv" in low or "loading" in low:
        return "loading"
    if "planning" in low or "provided score" in low:
        return "planning"
    if "generating song" in low:
        return "semantic"
    if "synthesizing" in low:
        return "synthesis"
    if "decoding" in low:
        return "decoding"
    return "loading"


def _make_studio_pipeline(on_stage_event: Callable[[dict], None]):
    """Dynamically subclass YuE2Pipeline so MLX is only imported when generating."""
    from lyra.pipeline import YuE2Pipeline

    class StudioYuE2Pipeline(YuE2Pipeline):
        @contextmanager
        def _status(self, label, *, total=None, unit=None):
            with super()._status(label, total=total, unit=unit) as inner:
                proxy = _ProgressStageProxy(inner, label, unit, total, on_stage_event)
                yield proxy
                proxy._push(force=True, status="completed")

    return StudioYuE2Pipeline


_CUDA_GRAPH_PATCHED = False


def _ensure_cuda_graph_attention_backend() -> None:
    """On Windows PyTorch builds without compiled FlashAttention, route GraphAR 'auto' to cuDNN/SDPA."""
    global _CUDA_GRAPH_PATCHED
    if _CUDA_GRAPH_PATCHED:
        return
    import torch
    import torch.nn.functional as F
    from yue2 import cuda_graph

    if not torch.cuda.is_available():
        _CUDA_GRAPH_PATCHED = True
        return

    flash_ok = False
    if hasattr(torch.ops.aten, "_flash_attention_forward"):
        try:
            q = torch.zeros((1, 16, 128), device="cuda", dtype=torch.bfloat16)
            k = torch.zeros((8, 8, 128), device="cuda", dtype=torch.bfloat16)
            v = torch.zeros((8, 8, 128), device="cuda", dtype=torch.bfloat16)
            cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
            cu_k = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
            used = torch.tensor([4], dtype=torch.int32, device="cuda")
            torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, 1, 8, 0.0, False, False, seqused_k=used)
            torch.cuda.synchronize()
            flash_ok = True
        except Exception:
            flash_ok = False

    if not flash_ok:
        fallback = "sdpa"
        if torch.backends.cudnn.is_available():
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                q = torch.zeros((1, 16, 1, 128), device="cuda", dtype=torch.bfloat16)
                k = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.bfloat16)
                v = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.bfloat16)
                mask = torch.ones((1, 1, 1, 8), device="cuda", dtype=torch.bool)
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True)
                torch.cuda.synchronize()
                fallback = "cudnn"
            except Exception:
                fallback = "sdpa"

        orig_init = cuda_graph.GraphAR.__init__

        def _patched_init(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto", fuse_projections=False):
            if attention_backend == "auto":
                attention_backend = fallback
            return orig_init(
                self,
                model,
                prefixes,
                max_tokens,
                capture=capture,
                attention_backend=attention_backend,
                fuse_projections=fuse_projections,
            )

        cuda_graph.GraphAR.__init__ = _patched_init

    _CUDA_GRAPH_PATCHED = True


def _make_torch_studio_pipeline(on_stage_event: Callable[[dict], None]):
    """Dynamically subclass upstream PyTorch/CUDA YuE2Pipeline for progress streaming."""
    _ensure_cuda_graph_attention_backend()
    from yue2.pipeline import YuE2Pipeline

    class TorchStudioYuE2Pipeline(YuE2Pipeline):
        @contextmanager
        def _status(self, label, *, total=None, unit=None):
            with super()._status(label, total=total, unit=unit) as inner:
                proxy = _ProgressStageProxy(inner, label, unit, total, on_stage_event)
                yield proxy
                proxy._push(force=True, status="completed")

    return TorchStudioYuE2Pipeline


def _read_song_entry(song_dir: Path) -> dict | None:
    audio_path = song_dir / "audio.flac"
    if not audio_path.is_file():
        return None

    result_data: dict[str, Any] = {}
    request_data: dict[str, Any] = {}
    config_data: dict[str, Any] = {}
    studio_meta: dict[str, Any] = {}

    for name, target in (
        ("result.json", result_data),
        ("request.json", request_data),
        ("config.json", config_data),
        ("studio_meta.json", studio_meta),
    ):
        p = song_dir / name
        if p.is_file():
            try:
                target.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    created_at = studio_meta.get("created_at")
    if not created_at:
        mtime = audio_path.stat().st_mtime
        created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    gen_cfg = config_data.get("generation") or {}
    timing = result_data.get("timing") or {}
    score_path = song_dir / "score.abc"
    has_abc = score_path.is_file() and score_path.stat().st_size > 0

    return {
        "id": song_dir.name,
        "title": studio_meta.get("title") or request_data.get("id") or song_dir.name,
        "style": request_data.get("style", ""),
        "lyrics": request_data.get("lyrics", ""),
        "cot": request_data.get("cot") or config_data.get("cot") or "full",
        "seed": request_data.get("seed"),
        "instrumental": bool(studio_meta.get("instrumental", False)),
        "precision": config_data.get("ar_precision") or studio_meta.get("precision") or "8bit",
        "ode_steps": gen_cfg.get("ode_steps") or studio_meta.get("ode_steps") or 8,
        "sample_rate": result_data.get("sample_rate", 48000),
        "audio_seconds": round(float(result_data.get("audio_seconds", 0.0)), 2),
        "e2e_seconds": round(float(timing.get("e2e_seconds", 0.0)), 1),
        "timing": timing,
        "truncated": result_data.get("truncated") or {},
        "has_abc": has_abc,
        "file_size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2),
        "created_at": created_at,
        "audio_url": f"/api/songs/{song_dir.name}/audio",
    }


def list_songs(outputs_dir: Path = OUTPUTS_DIR) -> list[dict]:
    if not outputs_dir.exists():
        return []
    items: list[dict] = []
    for child in outputs_dir.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            entry = _read_song_entry(child)
            if entry is not None:
                items.append(entry)
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def get_song_detail(song_id: str, outputs_dir: Path = OUTPUTS_DIR) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", song_id) or song_id in {".", ".."}:
        return None
    song_dir = (outputs_dir / song_id).resolve()
    if song_dir.parent != outputs_dir.resolve() or not song_dir.is_dir():
        return None
    entry = _read_song_entry(song_dir)
    if entry is None:
        return None
    score_path = song_dir / "score.abc"
    entry["abc"] = score_path.read_text(encoding="utf-8") if score_path.is_file() else None
    return entry


def delete_song(song_id: str, outputs_dir: Path = OUTPUTS_DIR) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", song_id) or song_id in {".", ".."}:
        return False
    song_dir = (outputs_dir / song_id).resolve()
    if song_dir.parent != outputs_dir.resolve() or not song_dir.is_dir():
        return False
    shutil.rmtree(song_dir)
    return True


@dataclass
class JobState:
    job_id: str = ""
    song_id: str = ""
    title: str = ""
    status: str = "idle"  # idle | running | completed | cancelled | failed
    stage: str = "idle"  # idle | loading | planning | semantic | synthesis | decoding | saving | completed
    stage_label: str = ""
    completed: int = 0
    total: int | None = None
    unit: str | None = None
    rate: float = 0.0
    estimated_audio_seconds: float | None = None
    overall_percent: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    song: dict | None = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = round(max(0.0, (self.finished_at or time.time()) - self.started_at), 1)
        return {
            "job_id": self.job_id,
            "song_id": self.song_id,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "rate": self.rate,
            "estimated_audio_seconds": self.estimated_audio_seconds,
            "overall_percent": round(min(100.0, max(0.0, self.overall_percent)), 1),
            "elapsed_seconds": elapsed,
            "error": self.error,
            "song": self.song,
            "params": self.params,
        }


class GenerationManager:
    """Manages a single serial YuE2 MLX generation worker with live progress & cancellation."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        vae_dir: Path = DEFAULT_VAE_DIR,
        outputs_dir: Path = OUTPUTS_DIR,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.model_dir = model_dir
        self.vae_dir = vae_dir
        self.outputs_dir = outputs_dir
        self._on_event = on_event
        self._lock = threading.Lock()
        self._state = JobState()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return self._state.to_dict()

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event_type, payload)
            except Exception:
                pass

    def is_running(self) -> bool:
        with self._lock:
            return self._state.status == "running"

    def cancel(self) -> dict:
        with self._lock:
            if self._state.status == "running":
                self._cancel_event.set()
                self._state.stage_label = "Cancelling generation..."
            snap = self._state.to_dict()
        self._emit("job", snap)
        return snap

    def start(self, params: GenerateParams) -> dict:
        model_info = inspect_models(self.model_dir, self.vae_dir)
        if not model_info["ready"]:
            raise RuntimeError(
                "YuE2-3B weights are not downloaded yet. Click 'Download Weights' first."
            )

        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("A song is already being generated. Wait for it to finish or cancel it first.")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            short = uuid.uuid4().hex[:6]
            song_id = f"{ts}-{_slugify(params.title or 'track')}-{short}"
            seed = params.seed if params.seed is not None else random.randint(1, 2**31 - 1)
            params.seed = seed

            self._cancel_event = threading.Event()
            init_label = (
                "Initializing PyTorch CUDA pipeline..."
                if BACKEND == "torch"
                else "Initializing Apple Silicon MLX pipeline..."
            )
            self._state = JobState(
                job_id=uuid.uuid4().hex[:12],
                song_id=song_id,
                title=params.title.strip() or "Untitled Track",
                status="running",
                stage="loading",
                stage_label=init_label,
                overall_percent=2.0,
                started_at=time.time(),
                params={
                    "cot": params.cot,
                    "precision": params.precision,
                    "ode_steps": params.ode_steps,
                    "max_tokens": params.max_tokens,
                    "seed": seed,
                    "instrumental": params.instrumental,
                },
            )
            self._worker_thread = threading.Thread(
                target=self._run_job,
                args=(params, song_id),
                name=f"yue2-gen-{song_id}",
                daemon=True,
            )
            self._worker_thread.start()
            snap = self._state.to_dict()

        self._emit("job", snap)
        return snap

    def _handle_stage_update(self, info: dict, params: GenerateParams) -> None:
        stage_key = _stage_key_for_label(info["label"])
        completed = info["completed"]
        total = info["total"]

        # Smooth overall progress approximation across the 5 stages:
        # loading: 0-8%, planning: 8-22%, semantic: 22-55%, synthesis: 55-90%, decoding: 90-98%
        pct = 5.0
        est_audio = None
        if stage_key == "loading":
            pct = 6.0
        elif stage_key == "planning":
            frac = min(1.0, completed / 600.0)
            pct = 8.0 + 14.0 * frac
        elif stage_key == "semantic":
            denom = max(300, params.max_tokens)
            frac = min(1.0, completed / float(denom))
            pct = 22.0 + 33.0 * frac
            est_audio = round(completed / 25.0, 1)
        elif stage_key == "synthesis":
            denom = total or params.ode_steps or 8
            frac = min(1.0, completed / float(max(1, denom)))
            pct = 55.0 + 35.0 * frac
        elif stage_key == "decoding":
            denom = total or 4
            frac = min(1.0, completed / float(max(1, denom)))
            pct = 90.0 + 8.0 * frac

        with self._lock:
            if self._state.status != "running":
                return
            self._state.stage = stage_key
            self._state.stage_label = info["label"]
            self._state.completed = completed
            self._state.total = total
            self._state.unit = info["unit"]
            self._state.rate = info["rate"]
            if est_audio is not None:
                self._state.estimated_audio_seconds = est_audio
            self._state.overall_percent = max(self._state.overall_percent, pct)
            snap = self._state.to_dict()
        self._emit("job", snap)

    def _run_job(self, params: GenerateParams, song_id: str) -> None:
        destination = self.outputs_dir / song_id
        effective_precision = params.precision
        try:
            from yue2.protocol import GenerationConfig, Sampling
            from yue2.storage import identity

            style = params.style.strip()
            lyrics = params.lyrics.strip()
            if params.instrumental:
                style = _format_style_instrumental(style)
                if not lyrics:
                    lyrics = "[Intro]\n\n[Instrumental]\n\n[Outro]"

            budget = params.memory_budget_gib or compute_safe_memory_budget_gib()
            semantic_sampling = Sampling(
                temperature=float(params.temperature),
                top_p=float(params.top_p),
                top_k=int(params.top_k),
                repetition_penalty=float(params.repetition_penalty),
                penalty_window=50,
                min_tokens=int(min(params.min_tokens, params.max_tokens)),
                max_tokens=int(params.max_tokens),
            )
            gen_config = GenerationConfig(
                semantic=semantic_sampling,
                ode_steps=int(params.ode_steps),
            )

            request_kwargs: dict[str, Any] = {
                "style": style,
                "lyrics": lyrics,
                "cot": params.cot,
                "seed": int(params.seed),
                "id": song_id,
            }
            if params.abc and params.abc.strip() and params.cot != "off":
                request_kwargs["abc"] = params.abc.strip()
            if params.cfg_scale is not None:
                request_kwargs["cfg_scale"] = float(params.cfg_scale)

            cancelled_fn = self._cancel_event.is_set

            if BACKEND == "torch":
                import torch
                from yue2.pipeline import SongResult

                torch_backend = os.environ.get("STUDIO_TORCH_BACKEND", "torch")
                can_fp8 = (
                    torch.cuda.is_available()
                    and torch.cuda.get_device_capability(0) >= (8, 9)
                )
                torch_quant = "fp8" if (params.precision == "fp8" and can_fp8) else "none"
                effective_precision = "fp8" if torch_quant == "fp8" else "bf16"
                offload_ar = budget <= 16.0
                vae_core_frames = 256 if budget <= 12.5 else 512

                StudioPipe = _make_torch_studio_pipeline(
                    lambda info: self._handle_stage_update(info, params)
                )
                with StudioPipe.from_pretrained(
                    str(self.model_dir),
                    vae=str(self.vae_dir),
                    device="auto",
                    memory_budget_gib=budget,
                    backend=torch_backend,
                    quantization=torch_quant,
                    offload_ar=offload_ar,
                    local_files_only=True,
                    generation_config=gen_config,
                    vae_core_frames=vae_core_frames,
                    progress=True,
                ) as pipe:
                    request = pipe._request(**request_kwargs)
                    effective_cfg = pipe.effective_config(request)
                    request_id = identity(
                        {"request": request.to_dict(), "config": effective_cfg, "weights": pipe.weights}
                    )

                    t0 = time.perf_counter()
                    plan = pipe.plan(request=request, cancelled=cancelled_fn)
                    semantic = pipe.generate_semantic(plan, cancelled=cancelled_fn)

                    with self._lock:
                        self._state.estimated_audio_seconds = round(len(semantic.tokens) / 25.0, 1)

                    nar_start = time.perf_counter()
                    latents = pipe.synthesize(semantic, cancelled=cancelled_fn)
                    nar_seconds = time.perf_counter() - nar_start

                    if cancelled_fn():
                        raise InterruptedError("Generation cancelled by user")

                    vae_start = time.perf_counter()
                    audio = pipe.decode(latents)
                    vae_seconds = time.perf_counter() - vae_start

                    timing = {
                        "abc": plan.timing,
                        "semantic": semantic.timing,
                        "nar_seconds": nar_seconds,
                        "vae_seconds": vae_seconds,
                        "load": dict(pipe.load_timing),
                        "e2e_seconds": time.perf_counter() - t0,
                    }
                    result = SongResult(
                        audio,
                        48000,
                        semantic,
                        latents,
                        effective_cfg,
                        pipe.weights,
                        timing,
                        request_id,
                    )
            else:
                import mlx.core as mx
                from lyra.pipeline import SongResult, initial_noise
                from .memory_guard import install_16gb_friendly_guard

                install_16gb_friendly_guard()

                StudioPipe = _make_studio_pipeline(
                    lambda info: self._handle_stage_update(info, params)
                )

                with StudioPipe.from_pretrained(
                    str(self.model_dir),
                    vae=str(self.vae_dir),
                    precision=params.precision,
                    local_files_only=True,
                    memory_budget_gib=budget,
                    vae_core_frames=256,
                    progress=True,
                    generation_config=gen_config,
                    require_ac=False,
                ) as pipe:
                    request = pipe._request(**request_kwargs)
                    effective_cfg = pipe.effective_config(request)
                    request_id = identity(
                        {"request": request.to_dict(), "config": effective_cfg, "weights": pipe.weights}
                    )

                    t0 = time.perf_counter()
                    plan = pipe.plan(request=request, cancelled=cancelled_fn)
                    semantic = pipe.generate_semantic(plan, cancelled=cancelled_fn)

                    # Update estimated final length once semantic tokens are complete
                    with self._lock:
                        self._state.estimated_audio_seconds = round(len(semantic.tokens) / 25.0, 1)

                    # Free 8-bit AR & KV cache before loading BF16 conditioning + NAR (~7.3 GB) on 16 GB Macs
                    if params.precision != "bf16" and getattr(pipe, "_ar", None) is not None:
                        pipe._ar = None
                        gc.collect()
                        mx.synchronize()
                        mx.clear_cache()

                    noise = initial_noise(len(semantic.tokens), request.seed)
                    nar_start = time.perf_counter()
                    latents = pipe.synthesize(semantic, cancelled=cancelled_fn, noise=noise)
                    nar_seconds = time.perf_counter() - nar_start

                    if cancelled_fn():
                        raise InterruptedError("Generation cancelled by user")

                    # Critical optimization for 16 GB Apple M1: release AR + NAR (~7.3 GB) before loading 48kHz FP32 VAE!
                    pipe._release_models()
                    gc.collect()
                    mx.clear_cache()

                    vae_start = time.perf_counter()
                    audio = pipe.decode(latents, cancelled=cancelled_fn)
                    vae_seconds = time.perf_counter() - vae_start

                    timing = {
                        "abc": plan.timing,
                        "semantic": semantic.timing,
                        "nar_seconds": nar_seconds,
                        "vae_seconds": vae_seconds,
                        "load": dict(pipe.load_timing),
                        "e2e_seconds": time.perf_counter() - t0,
                    }
                    result = SongResult(
                        audio,
                        48000,
                        semantic,
                        latents,
                        effective_cfg,
                        pipe.weights,
                        timing,
                        request_id,
                        noise,
                    )

            # Save artifacts outside GPUExecution context so disk I/O never fights the memory guard
            with self._lock:
                self._state.stage = "saving"
                self._state.stage_label = "Saving 48kHz lossless FLAC & ABC score..."
                self._state.overall_percent = 99.0
            self._emit("job", self.snapshot())

            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            result.save_artifacts(destination)

            studio_meta = {
                "title": params.title.strip() or "Untitled Track",
                "instrumental": bool(params.instrumental),
                "precision": effective_precision,
                "ode_steps": int(params.ode_steps),
                "max_tokens": int(params.max_tokens),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (destination / "studio_meta.json").write_text(
                json.dumps(studio_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            song_entry = get_song_detail(song_id, self.outputs_dir)
            with self._lock:
                self._state.status = "completed"
                self._state.stage = "completed"
                self._state.stage_label = "Song ready!"
                self._state.overall_percent = 100.0
                self._state.finished_at = time.time()
                self._state.song = song_entry
                snap = self._state.to_dict()
            self._emit("job", snap)
            self._emit("songs_updated", {"song_id": song_id})

        except InterruptedError:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            with self._lock:
                self._state.status = "cancelled"
                self._state.stage = "idle"
                self._state.stage_label = "Generation cancelled"
                self._state.finished_at = time.time()
                snap = self._state.to_dict()
            self._emit("job", snap)

        except Exception as exc:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            msg = str(exc)
            if isinstance(exc, MemoryError) or "memory" in msg.lower() or "swapping" in msg.lower() or "out of memory" in msg.lower():
                if BACKEND == "torch":
                    msg = (
                        f"{msg}. Tip for 12 GB NVIDIA GPU: close other GPU-heavy apps/browsers, "
                        "use 8 ODE steps, and shorter max tokens (600–1200)."
                    )
                else:
                    msg = (
                        f"{msg}. Tip for 16 GB Apple M1: close heavy browser tabs/apps, "
                        "use 8-bit precision, 8 ODE steps, and shorter max tokens (600–1200)."
                    )
            with self._lock:
                self._state.status = "failed"
                self._state.stage = "idle"
                self._state.stage_label = "Generation failed"
                self._state.error = msg
                self._state.finished_at = time.time()
                snap = self._state.to_dict()
            self._emit("job", snap)

        finally:
            gc.collect()
            if BACKEND == "torch":
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass
