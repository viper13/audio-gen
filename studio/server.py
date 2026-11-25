"""FastAPI server for YuE2 MLX Studio (Generation + Audio Analysis + SSE + Web UI)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

os.environ.setdefault("MLX_ENABLE_TF32", "0")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analyzer import (
    ALLOWED_AUDIO_EXTS,
    ANALYSES_DIR,
    AnalysisManager,
    check_analysis_dependencies,
    delete_analysis,
    get_analysis_detail,
    list_analyses,
)
from .config import DEFAULT_MODEL_DIR, DEFAULT_VAE_DIR, OUTPUTS_DIR, STATIC_DIR, TMP_DIR, get_system_status
from .generator import (
    GenerateParams,
    GenerationManager,
    delete_song,
    get_song_detail,
    list_songs,
)
from .model_manager import ModelDownloadManager
from .presets import SONG_PRESETS, STYLE_TAG_GROUPS


class EventBroker:
    """Broadcasts thread-safe events from worker threads to async SSE clients."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[tuple[str, dict]]] = set()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event_type: str, data: dict) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _enqueue() -> None:
            for q in list(self._subscribers):
                try:
                    q.put_nowait((event_type, data))
                except asyncio.QueueFull:
                    pass

        loop.call_soon_threadsafe(_enqueue)

    def subscribe(self) -> asyncio.Queue[tuple[str, dict]]:
        q: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=128)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[tuple[str, dict]]) -> None:
        self._subscribers.discard(q)


class GenerateRequest(BaseModel):
    title: str = Field(default="Untitled Track", max_length=120)
    style: str = Field(..., min_length=2, max_length=2000)
    lyrics: str = Field(default="", max_length=12000)
    cot: Literal["full", "melody", "off"] = "melody"
    precision: Literal["8bit", "bf16", "4bit", "fp8"] = "8bit"
    ode_steps: int = Field(default=8, ge=1, le=64)
    max_tokens: int = Field(default=1200, ge=200, le=9000)
    min_tokens: int = Field(default=200, ge=0, le=9000)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    instrumental: bool = False
    abc: str | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=2.5)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=100, ge=1, le=500)
    repetition_penalty: float = Field(default=1.2, gt=0.0, le=3.0)
    cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)


broker = EventBroker()
model_manager = ModelDownloadManager(
    model_dir=DEFAULT_MODEL_DIR,
    vae_dir=DEFAULT_VAE_DIR,
    on_change=lambda snap: broker.publish("models", snap),
)
gen_manager = GenerationManager(
    model_dir=DEFAULT_MODEL_DIR,
    vae_dir=DEFAULT_VAE_DIR,
    outputs_dir=OUTPUTS_DIR,
    on_event=lambda etype, data: broker.publish(etype, data),
)
analysis_manager = AnalysisManager(
    analyses_dir=ANALYSES_DIR,
    on_event=lambda etype, data: broker.publish(etype, data),
)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    broker.attach_loop(asyncio.get_running_loop())
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="YuE2 MLX Studio", version="1.1.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/system")
async def api_system() -> dict:
    status = get_system_status()
    status["analysis"] = check_analysis_dependencies()
    return status


@app.get("/api/presets")
async def api_presets() -> dict:
    return {"tag_groups": STYLE_TAG_GROUPS, "presets": SONG_PRESETS}


@app.get("/api/models")
async def api_models() -> dict:
    return model_manager.snapshot()


@app.post("/api/models/download")
async def api_models_download() -> dict:
    return model_manager.start_download()


# --- Generation Endpoints ---
@app.get("/api/jobs/current")
async def api_job_current() -> dict:
    return gen_manager.snapshot()


@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> dict:
    if analysis_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="An audio analysis job is currently using the GPU. Wait for it to finish or cancel it first.",
        )
    if not req.instrumental and not req.lyrics.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter song lyrics or enable Instrumental mode.",
        )
    params = GenerateParams(
        title=req.title,
        style=req.style,
        lyrics=req.lyrics,
        cot=req.cot,
        precision=req.precision,
        ode_steps=req.ode_steps,
        max_tokens=req.max_tokens,
        min_tokens=min(req.min_tokens, req.max_tokens),
        seed=req.seed,
        instrumental=req.instrumental,
        abc=req.abc,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        cfg_scale=req.cfg_scale,
    )
    try:
        return gen_manager.start(params)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/cancel")
async def api_job_cancel() -> dict:
    return gen_manager.cancel()


@app.get("/api/songs")
async def api_songs() -> dict:
    return {"songs": list_songs(OUTPUTS_DIR)}


@app.get("/api/songs/{song_id}")
async def api_song_detail(song_id: str) -> dict:
    detail = get_song_detail(song_id, OUTPUTS_DIR)
    if detail is None:
        raise HTTPException(status_code=404, detail="Song not found")
    return detail


@app.get("/api/songs/{song_id}/audio")
async def api_song_audio(song_id: str) -> FileResponse:
    detail = get_song_detail(song_id, OUTPUTS_DIR)
    if detail is None:
        raise HTTPException(status_code=404, detail="Song not found")
    audio_path = OUTPUTS_DIR / song_id / "audio.flac"
    return FileResponse(audio_path, media_type="audio/flac", filename=f"{song_id}.flac")


@app.delete("/api/songs/{song_id}")
async def api_song_delete(song_id: str) -> dict:
    if not delete_song(song_id, OUTPUTS_DIR):
        raise HTTPException(status_code=404, detail="Song not found")
    broker.publish("songs_updated", {"deleted": song_id})
    return {"deleted": song_id}


# --- Audio Analysis & Transcription Endpoints ---
@app.get("/api/analyses")
async def api_analyses() -> dict:
    return {"analyses": list_analyses(ANALYSES_DIR)}


@app.get("/api/analyses/job")
async def api_analysis_job() -> dict:
    return analysis_manager.snapshot()


@app.post("/api/analyses/cancel")
async def api_analysis_cancel() -> dict:
    return analysis_manager.cancel()


@app.post("/api/analyses/upload")
async def api_analysis_upload(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    task: str = Form(default="full"),
    max_seconds: str = Form(default=""),
) -> dict:
    if gen_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A song generation job is currently using the GPU. Wait for it to finish or cancel it first.",
        )
    if task not in {"full", "melody-full", "melody-vocal"}:
        raise HTTPException(status_code=400, detail="Invalid analysis task")

    filename = file.filename or "audio.mp3"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format {ext!r}. Use MP3, WAV, FLAC, M4A, AAC, or OGG.",
        )

    parsed_max: float | None = None
    if max_seconds.strip():
        try:
            parsed_max = float(max_seconds.strip())
            if parsed_max <= 0:
                parsed_max = None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="max_seconds must be a positive number") from exc

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="yue2-upload-", suffix=ext, dir=str(TMP_DIR))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return analysis_manager.start(
            source_path=tmp_path,
            original_filename=filename,
            title=title.strip() or Path(filename).stem,
            task=task,
            max_seconds=parsed_max,
        )
    except RuntimeError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/analyses/{analysis_id}")
async def api_analysis_detail(analysis_id: str) -> dict:
    detail = get_analysis_detail(analysis_id, ANALYSES_DIR)
    if detail is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return detail


@app.get("/api/analyses/{analysis_id}/audio")
async def api_analysis_audio(analysis_id: str) -> FileResponse:
    detail = get_analysis_detail(analysis_id, ANALYSES_DIR)
    if detail is None or not detail.get("has_source_audio"):
        raise HTTPException(status_code=404, detail="Source audio not found")
    meta = json.loads((ANALYSES_DIR / analysis_id / "studio_meta.json").read_text(encoding="utf-8"))
    audio_path = ANALYSES_DIR / analysis_id / meta["source_audio_file"]
    return FileResponse(audio_path, filename=meta.get("original_filename") or audio_path.name)


@app.get("/api/analyses/{analysis_id}/file/{filename}")
async def api_analysis_file(analysis_id: str, filename: str):
    allowed = {
        "score.abc",
        "transcription.mid",
        "melody.mid",
        "melody_vocal.mid",
        "melody_instrumental.mid",
        "chord.lab",
        "key.lab",
        "structure.lab",
    }
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Invalid artifact filename")
    detail = get_analysis_detail(analysis_id, ANALYSES_DIR)
    if detail is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    path = ANALYSES_DIR / analysis_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not available for this analysis")
    if filename.endswith(".mid"):
        return FileResponse(path, media_type="audio/midi", filename=f"{analysis_id}-{filename}")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@app.delete("/api/analyses/{analysis_id}")
async def api_analysis_delete(analysis_id: str) -> dict:
    if not delete_analysis(analysis_id, ANALYSES_DIR):
        raise HTTPException(status_code=404, detail="Analysis not found")
    broker.publish("analyses_updated", {"deleted": analysis_id})
    return {"deleted": analysis_id}


# --- SSE Stream ---
@app.get("/api/events")
async def api_events() -> StreamingResponse:
    queue = broker.subscribe()

    async def event_stream():
        try:
            sys_snap = get_system_status()
            sys_snap["analysis"] = check_analysis_dependencies()
            yield f"event: job\ndata: {json.dumps(gen_manager.snapshot())}\n\n"
            yield f"event: analysis_job\ndata: {json.dumps(analysis_manager.snapshot())}\n\n"
            yield f"event: models\ndata: {json.dumps(model_manager.snapshot())}\n\n"
            yield f"event: system\ndata: {json.dumps(sys_snap)}\n\n"

            while True:
                try:
                    etype, data = await asyncio.wait_for(queue.get(), timeout=4.0)
                    yield f"event: {etype}\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    sys_snap = get_system_status()
                    sys_snap["analysis"] = check_analysis_dependencies()
                    yield f"event: system\ndata: {json.dumps(sys_snap)}\n\n"
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("STUDIO_PORT", "8765"))
    uvicorn.run("studio.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
