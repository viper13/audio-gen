#!/usr/bin/env python3
"""Quick CLI wrapper for YuE2-3B MLX music generation on Apple Silicon (16 GB M1 safe defaults)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["MLX_ENABLE_TF32"] = "0"

from studio.config import DEFAULT_MODEL_DIR, DEFAULT_VAE_DIR, OUTPUTS_DIR, compute_safe_memory_budget_gib
from studio.generator import GenerateParams, GenerationManager
from studio.model_manager import inspect_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a song locally with YuE2-3B (Apple Silicon MLX)")
    parser.add_argument("--title", default="CLI Track", help="Song title")
    parser.add_argument("--style", required=True, help="Genre/style prompt (e.g. 'indie pop, female vocal, 100 BPM')")
    parser.add_argument("--lyrics", default="", help="Song lyrics text")
    parser.add_argument("--lyrics-file", type=Path, help="Read lyrics from UTF-8 file")
    parser.add_argument("--instrumental", action="store_true", help="Generate instrumental track without vocals")
    parser.add_argument("--mode", choices=("full", "melody", "off"), default="melody", help="Score planning mode")
    parser.add_argument("--precision", choices=("8bit", "bf16"), default="8bit", help="AR quantization precision")
    parser.add_argument("--steps", type=int, default=8, help="Acoustic ODE steps (8=fast, 32=studio)")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max semantic tokens (25 tokens ≈ 1s)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    info = inspect_models(DEFAULT_MODEL_DIR, DEFAULT_VAE_DIR)
    if not info["ready"]:
        print(
            f"Error: Model weights not found at {DEFAULT_MODEL_DIR}.\n"
            "Download them via the Web UI or run:\n"
            "  engine/.venv/bin/python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('vanch007/mlx-Yue2-3B', local_dir='models/mlx-Yue2-3B')\"",
            file=sys.stderr,
        )
        return 1

    lyrics = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else args.lyrics
    if not args.instrumental and not lyrics.strip():
        print("Error: provide --lyrics, --lyrics-file, or --instrumental", file=sys.stderr)
        return 1

    params = GenerateParams(
        title=args.title,
        style=args.style,
        lyrics=lyrics,
        cot=args.mode,
        precision=args.precision,
        ode_steps=args.steps,
        max_tokens=args.max_tokens,
        seed=args.seed,
        instrumental=args.instrumental,
        memory_budget_gib=compute_safe_memory_budget_gib(),
    )

    done_state = {}

    def on_event(etype: str, data: dict) -> None:
        if etype == "job":
            done_state.clear()
            done_state.update(data)
            if data["status"] == "running":
                extra = f" ({data['completed']}/{data['total']} {data['unit']})" if data.get("total") else (
                    f" ({data['completed']} {data['unit']})" if data.get("completed") else ""
                )
                print(f"\r[{data['overall_percent']:5.1f}%] {data['stage_label']}{extra}   ", end="", flush=True)

    mgr = GenerationManager(DEFAULT_MODEL_DIR, DEFAULT_VAE_DIR, OUTPUTS_DIR, on_event=on_event)
    mgr.start(params)
    if mgr._worker_thread is not None:
        mgr._worker_thread.join()

    print()
    final = mgr.snapshot()
    if final["status"] == "completed" and final.get("song"):
        song = final["song"]
        out_path = OUTPUTS_DIR / song["id"] / "audio.flac"
        print(f"Done! Saved {song['audio_seconds']}s song to: {out_path}")
        return 0
    print(f"Failed ({final['status']}): {final.get('error')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
