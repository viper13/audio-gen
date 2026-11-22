# YuE2 MLX Studio (Apple Silicon)

Local, Suno-style AI full-song generation studio powered by **YuE2-3B** (HKUST / Multimodal Art Projection) running natively on **Apple Silicon via Apple MLX** with **8-bit quantization** — no PyTorch or CUDA required.

## Features

- **Native Apple Silicon (MLX) Engine**: Uses [`vanch007/mlx-Yue`](https://github.com/vanch007/mlx-Yue) (YuE2-3B + 48kHz FP32 Oobleck VAE) with Metal Steel SDPA kernels.
- **16 GB Mac Friendly**:
  - 8-bit quantized autoregressive model (`ar-8bit.safetensors`, ~2.66 GB).
  - Automatic memory budget sizing (`12.0 GiB` max on 16 GB Macs, leaving 4 GiB OS headroom).
  - Automatic AR/NAR unloading before 48kHz VAE decoding to minimize peak unified memory.
- **Suno-Style Web Studio UI**:
  - Genre / vocal / mood / instrument quick tag chips & multilingual presets (Ukrainian, English, Instrumental).
  - Lyrics editor with section insertion (`[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`).
  - Supports **Melody Lead**, **Full Auto ABC Score**, **Direct**, **Instrumental**, and **Custom ABC Score** modes.
  - Fast **8-step** acoustic synthesis (recommended for M1) up to **32-step** studio fidelity.
  - Real-time SSE progress stepper, live token/s counter, cancel button, and one-click Hugging Face weight downloader.
  - Persistent bottom **Web Audio waveform player**, song library, ABC score inspector, and 48kHz lossless `.flac` export.
- **Analyze & Cover Any Audio Track (SheetSage2 + MERT-v2 MLX)**:
  - Drag-and-drop any `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, or `.ogg` file in the **Analyze & Cover Audio** tab.
  - Extracts **BPM**, **musical key**, **time signature**, **song structure timeline**, **timestamped chord progression**, **vocal & instrumental melodies**, downloadable **MIDI** (`.mid`), **ABC score** (`.abc`), and **`.lab`** annotations.
  - One-click **Cover in YuE2** button loads the extracted ABC score & tempo directly into the Composer so you can remix or re-arrange the track into a new style.

## Quick Start

### 1. Launch the Web Studio

```bash
./start.sh
```

Open **http://127.0.0.1:8765** in your browser.

### 2. Download Model Weights (~10.6 GB, one-time)

Click **Download Weights** inside the Web UI, or run from terminal:

```bash
engine/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('vanch007/mlx-Yue2-3B', local_dir='models/mlx-Yue2-3B')"
```

### 3. CLI Generation (Optional)

```bash
engine/.venv/bin/python generate.py \
  --title "Midnight Neon" \
  --style "indie synthpop, expressive female vocal, warm analog synths, 108 BPM" \
  --lyrics "[Verse]\nNeon lights reflect in the rain\n[Chorus]\nStay with me tonight" \
  --precision 8bit \
  --steps 8 \
  --max-tokens 900
```

Generated songs (48kHz `.flac` audio, `.abc` score, and metadata) are saved under `outputs/<song_id>/`.

## Tips for Apple M1 (16 GB Unified Memory)

- Close heavy apps/browser tabs before generating so at least **5–6 GB** of RAM is available.
- Keep **AR Quantization = 8-bit** and **Acoustic Steps = 8 Steps (Fast)** for smooth, fast generation.
- Always run with `MLX_ENABLE_TF32=0` (handled automatically by `./start.sh`, `studio/`, and `generate.py`).
