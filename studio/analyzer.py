"""Native MLX SheetSage2 + MERT-v2-FullSong audio analyzer & library manager."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable
import uuid

os.environ.setdefault("MLX_ENABLE_TF32", "0")

from .config import BACKEND, HF_CACHE_DIR, MODELS_DIR, ROOT_DIR, compute_safe_memory_budget_gib

ANALYSES_DIR = ROOT_DIR / "analyses"
ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".webm"}


def _slugify(text: str, fallback: str = "audio") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return (cleaned[:40] or fallback).strip("-") or fallback


def _parse_lab_intervals(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            try:
                start = round(float(parts[0]), 2)
                end = round(float(parts[1]), 2)
                label = parts[2].strip()
                if label and end > start:
                    rows.append({"start": start, "end": end, "label": label})
            except ValueError:
                continue
    return rows


def _parse_beats_summary(path: Path) -> dict:
    if not path.is_file():
        return {"bpm": None, "meter": None, "beat_count": 0}
    times: list[float] = []
    meter_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            try:
                times.append(float(parts[0]))
                m = f"{int(parts[2])}/{int(parts[3])}"
                meter_counts[m] = meter_counts.get(m, 0) + 1
            except ValueError:
                continue
    bpm = None
    if len(times) >= 3:
        diffs = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        if diffs:
            med = diffs[len(diffs) // 2]
            if med > 0:
                bpm = round(60.0 / med, 1)
    meter = max(meter_counts.items(), key=lambda kv: kv[1])[0] if meter_counts else None
    return {"bpm": bpm, "meter": meter, "beat_count": len(times)}


def _extract_abc_header_info(abc_text: str | None) -> dict:
    info = {"bpm": None, "meter": None, "key": None}
    if not abc_text:
        return info
    for line in abc_text.splitlines()[:15]:
        if line.startswith("Q:1/4="):
            try:
                info["bpm"] = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("M:"):
            info["meter"] = line[2:].strip()
        elif line.startswith("K:"):
            info["key"] = line[2:].strip()
    return info


def check_analysis_dependencies() -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    missing_py = []
    for mod in ("mido", "pretty_midi", "scipy", "mir_eval"):
        try:
            __import__(mod)
        except ImportError:
            missing_py.append(mod)
    reason = None
    if not ffmpeg_path:
        reason = "FFmpeg was not found in PATH."
    elif missing_py:
        reason = f"Missing Python packages for audio analysis: {', '.join(missing_py)}."
    return {
        "ready": bool(ffmpeg_path) and len(missing_py) == 0,
        "supported": True,
        "backend": BACKEND,
        "reason": reason,
        "ffmpeg": ffmpeg_path,
        "missing_python": missing_py,
    }


def _read_analysis_entry(analysis_dir: Path, include_full: bool = False) -> dict | None:
    res_path = analysis_dir / "result.json"
    if not res_path.is_file():
        return None
    try:
        res = json.loads(res_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    meta: dict[str, Any] = {}
    meta_path = analysis_dir / "studio_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    abc_path = analysis_dir / "score.abc"
    abc_text = abc_path.read_text(encoding="utf-8") if abc_path.is_file() else None
    abc_hdr = _extract_abc_header_info(abc_text)
    beats = _parse_beats_summary(analysis_dir / "beat.lab")
    chords = _parse_lab_intervals(analysis_dir / "chord.lab")
    keys = _parse_lab_intervals(analysis_dir / "key.lab")
    structures = _parse_lab_intervals(analysis_dir / "structure.lab")

    primary_key = abc_hdr["key"] or (keys[0]["label"] if keys else None)
    bpm = abc_hdr["bpm"] or beats["bpm"]
    meter = abc_hdr["meter"] or beats["meter"]

    # Deduplicate consecutive chords for quick progression preview
    chord_seq: list[str] = []
    for c in chords:
        lbl = c["label"]
        if lbl in {"N", "X"}:
            continue
        if not chord_seq or chord_seq[-1] != lbl:
            chord_seq.append(lbl)

    created_at = meta.get("created_at") or datetime.fromtimestamp(
        res_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    midi_files = [
        name
        for name in ("transcription.mid", "melody.mid", "melody_vocal.mid", "melody_instrumental.mid")
        if (analysis_dir / name).is_file()
    ]
    source_audio = meta.get("source_audio_file")
    has_source_audio = bool(source_audio and (analysis_dir / source_audio).is_file())

    entry = {
        "id": analysis_dir.name,
        "title": meta.get("title") or analysis_dir.name,
        "source_filename": meta.get("original_filename") or source_audio or "",
        "task": res.get("task", "full"),
        "status": res.get("status", "complete"),
        "duration_seconds": round(float(res.get("duration_seconds", 0.0)), 2),
        "elapsed_seconds": round(float(res.get("elapsed_seconds", 0.0)), 1),
        "bpm": bpm,
        "meter": meter,
        "key": primary_key,
        "keys": keys,
        "structures": structures,
        "chord_count": len(chords),
        "chord_preview": chord_seq[:16],
        "melody_notes": res.get("melody_notes", 0),
        "vocal_notes": res.get("vocal_notes", 0),
        "instrumental_notes": res.get("instrumental_notes", 0),
        "abc_measures": res.get("abc_measures", 0),
        "has_abc": bool(abc_text),
        "midi_files": midi_files,
        "has_source_audio": has_source_audio,
        "audio_url": f"/api/analyses/{analysis_dir.name}/audio" if has_source_audio else None,
        "created_at": created_at,
    }
    if include_full:
        entry["abc"] = abc_text
        entry["chords"] = chords
        entry["warnings"] = res.get("warnings") or []
    return entry


def list_analyses(analyses_dir: Path = ANALYSES_DIR) -> list[dict]:
    if not analyses_dir.exists():
        return []
    items = []
    for child in analyses_dir.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            entry = _read_analysis_entry(child, include_full=False)
            if entry is not None:
                items.append(entry)
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def get_analysis_detail(analysis_id: str, analyses_dir: Path = ANALYSES_DIR) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", analysis_id) or analysis_id in {".", ".."}:
        return None
    target = (analyses_dir / analysis_id).resolve()
    if target.parent != analyses_dir.resolve() or not target.is_dir():
        return None
    return _read_analysis_entry(target, include_full=True)


def delete_analysis(analysis_id: str, analyses_dir: Path = ANALYSES_DIR) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", analysis_id) or analysis_id in {".", ".."}:
        return False
    target = (analyses_dir / analysis_id).resolve()
    if target.parent != analyses_dir.resolve() or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


@dataclass
class AnalysisJobState:
    job_id: str = ""
    analysis_id: str = ""
    title: str = ""
    status: str = "idle"  # idle | running | completed | cancelled | failed
    stage: str = "idle"
    stage_label: str = ""
    window: int = 0
    windows: int = 0
    tokens: int = 0
    percent: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None

    def to_dict(self) -> dict:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = round(max(0.0, (self.finished_at or time.time()) - self.started_at), 1)
        return {
            "job_id": self.job_id,
            "analysis_id": self.analysis_id,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "window": self.window,
            "windows": self.windows,
            "tokens": self.tokens,
            "percent": round(min(100.0, max(0.0, self.percent)), 1),
            "elapsed_seconds": elapsed,
            "error": self.error,
            "result": self.result,
        }


class AnalysisManager:
    """Runs SheetSage2 + MERT2 MLX audio analysis serially with cancellation & progress events."""

    def __init__(
        self,
        analyses_dir: Path = ANALYSES_DIR,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.analyses_dir = analyses_dir
        self._on_event = on_event
        self._lock = threading.Lock()
        self._state = AnalysisJobState()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return self._state.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._state.status == "running"

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event_type, payload)
            except Exception:
                pass

    def cancel(self) -> dict:
        with self._lock:
            if self._state.status == "running":
                self._cancel_event.set()
                self._state.stage_label = "Cancelling analysis..."
            snap = self._state.to_dict()
        self._emit("analysis_job", snap)
        return snap

    def start(
        self,
        *,
        source_path: Path,
        original_filename: str,
        title: str | None = None,
        task: str = "full",
        max_seconds: float | None = None,
    ) -> dict:
        deps = check_analysis_dependencies()
        if not deps["ready"]:
            raise RuntimeError(
                f"Audio analysis dependencies missing (ffmpeg={bool(deps['ffmpeg'])}, "
                f"missing python={deps['missing_python']})."
            )

        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("An audio analysis job is already running.")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            short = uuid.uuid4().hex[:6]
            clean_title = (title or Path(original_filename).stem or "Audio Track").strip()
            analysis_id = f"{ts}-{_slugify(clean_title)}-{short}"

            backend_label = "PyTorch CUDA" if BACKEND == "torch" else "MLX"
            self._cancel_event = threading.Event()
            self._state = AnalysisJobState(
                job_id=uuid.uuid4().hex[:12],
                analysis_id=analysis_id,
                title=clean_title,
                status="running",
                stage="loading",
                stage_label=f"Loading SheetSage2 + MERT-v2-FullSong ({backend_label}, downloads on first run)...",
                percent=5.0,
                started_at=time.time(),
            )
            self._thread = threading.Thread(
                target=self._worker,
                args=(analysis_id, source_path, original_filename, clean_title, task, max_seconds),
                name=f"yue2-analyze-{analysis_id}",
                daemon=True,
            )
            self._thread.start()
            snap = self._state.to_dict()

        self._emit("analysis_job", snap)
        return snap

    def _worker(
        self,
        analysis_id: str,
        temp_source: Path,
        original_filename: str,
        title: str,
        task: str,
        max_seconds: float | None,
    ) -> None:
        dest = self.analyses_dir / analysis_id
        try:
            self.analyses_dir.mkdir(parents=True, exist_ok=True)
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (HF_CACHE_DIR / "hub").mkdir(parents=True, exist_ok=True)
            budget = compute_safe_memory_budget_gib()

            def on_progress(evt: dict) -> None:
                with self._lock:
                    if self._state.status != "running":
                        return
                    stage = evt.get("stage", "")
                    if stage == "encoding":
                        win = int(evt.get("window", 1))
                        total_w = max(1, int(evt.get("windows", 1)))
                        self._state.stage = "encoding"
                        self._state.window = win
                        self._state.windows = total_w
                        self._state.stage_label = f"Analyzing audio window {win}/{total_w} with MERT-v2..."
                        self._state.percent = 15.0 + 75.0 * ((win - 1) / total_w)
                    elif stage == "decoding":
                        toks = int(evt.get("tokens", 0))
                        win = max(1, self._state.window)
                        total_w = max(1, self._state.windows)
                        self._state.stage = "decoding"
                        self._state.tokens = toks
                        self._state.stage_label = (
                            f"Transcribing chords, key, structure & melody "
                            f"(window {win}/{total_w} · {toks} events)..."
                        )
                        win_base = 15.0 + 75.0 * ((win - 1) / total_w)
                        win_span = 75.0 / total_w
                        self._state.percent = min(94.0, win_base + win_span * min(0.92, toks / 1200.0))
                    snap = self._state.to_dict()
                self._emit("analysis_job", snap)

            if BACKEND == "torch":
                from .torch_transcription import transcribe as torch_transcribe

                torch_transcribe(
                    temp_source,
                    dest,
                    offline=False,
                    cache_dir=str(HF_CACHE_DIR / "hub"),
                    device="auto",
                    task=task,
                    max_seconds=max_seconds,
                    cancelled=self._cancel_event.is_set,
                    progress=on_progress,
                )
            else:
                from lyra.measure import GPUExecution
                from lyra.transcription.pipeline import transcribe
                from .memory_guard import install_16gb_friendly_guard

                install_16gb_friendly_guard()
                with GPUExecution(backend="mlx", memory_budget_gib=budget) as guard:
                    def cancelled() -> bool:
                        guard.check()
                        return self._cancel_event.is_set()

                    transcribe(
                        temp_source,
                        dest,
                        offline=False,
                        cache_dir=str(HF_CACHE_DIR / "hub"),
                        task=task,
                        max_seconds=max_seconds,
                        cancelled=cancelled,
                        progress=on_progress,
                    )
                    guard.check()

            # Preserve the uploaded source audio inside the analysis directory for playback
            ext = Path(original_filename).suffix.lower() or ".audio"
            saved_audio_name = f"source{ext}"
            shutil.copy2(temp_source, dest / saved_audio_name)

            meta = {
                "title": title,
                "original_filename": original_filename,
                "source_audio_file": saved_audio_name,
                "task": task,
                "max_seconds": max_seconds,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (dest / "studio_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            detail = get_analysis_detail(analysis_id, self.analyses_dir)
            with self._lock:
                self._state.status = "completed"
                self._state.stage = "completed"
                self._state.stage_label = "Analysis complete!"
                self._state.percent = 100.0
                self._state.finished_at = time.time()
                self._state.result = detail
                snap = self._state.to_dict()
            self._emit("analysis_job", snap)
            self._emit("analyses_updated", {"analysis_id": analysis_id})

        except InterruptedError:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            with self._lock:
                self._state.status = "cancelled"
                self._state.stage = "idle"
                self._state.stage_label = "Analysis cancelled"
                self._state.finished_at = time.time()
                snap = self._state.to_dict()
            self._emit("analysis_job", snap)

        except Exception as exc:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            with self._lock:
                self._state.status = "failed"
                self._state.stage = "idle"
                self._state.stage_label = "Analysis failed"
                self._state.error = str(exc)
                self._state.finished_at = time.time()
                snap = self._state.to_dict()
            self._emit("analysis_job", snap)

        finally:
            try:
                temp_source.unlink(missing_ok=True)
            except Exception:
                pass
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
