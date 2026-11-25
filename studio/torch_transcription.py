"""Native PyTorch/CUDA SheetSage2 + MERT-v2-FullSong audio transcription pipeline.

Ported from engine/src/lyra/{mert,bart,transcription/{model,pipeline}}.py to run
natively on NVIDIA CUDA (or CPU) while reusing the exact pinned upstream
SheetSage2 tokenizer, grammar constraints, sliding windows, and ABC/MIDI exports.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Callable

import numpy as np
from safetensors.numpy import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F

from lyra.transcription.control import check_cancelled
from lyra.transcription.upstream.exports_sheetsage2 import export_result
from lyra.transcription.upstream.generation_sheetsage2 import (
    PromptGrammarState,
    build_overlap_prefix_tokens,
    decode_generated_tokens,
    event_time_map,
    stitched_window_events,
)
from lyra.transcription.upstream.tokenization_sheetsage2 import SheetSage2Tokenizer
from lyra.transcription.upstream.windows import sliding_window_plan
from yue2.storage import sha256_file, write_json

MODEL_REPO = "m-a-p/SheetSage2"
MODEL_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"


# ---------------------------------------------------------------------------
# MERT-v2-FullSong Encoder (official checkpoint layout)
# ---------------------------------------------------------------------------


class _Transpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class _SpectrogramBuffers(nn.Module):
    def __init__(self, win_length: int):
        super().__init__()
        self.register_buffer("window", torch.zeros(win_length, dtype=torch.float32))


class _MelScaleBuffers(nn.Module):
    def __init__(self, n_freqs: int, n_mels: int):
        super().__init__()
        self.register_buffer("fb", torch.zeros((n_freqs, n_mels), dtype=torch.float32))


class MelFrontend(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_fft = int(config["n_fft"])
        self.hop = int(config["hop_length"])
        bins = int(config["num_mel_bins"])
        self.register_buffer("mel_mean", torch.zeros(bins, dtype=torch.float32))
        self.register_buffer("mel_std", torch.ones(bins, dtype=torch.float32))
        self.spectrogram = _SpectrogramBuffers(int(config["win_length"]))
        self.mel_scale = _MelScaleBuffers(self.n_fft // 2 + 1, bins)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Matches torchaudio Spectrogram defaults: centered reflect padding,
        # periodic Hann window, unnormalized power spectrum.
        x = F.pad(waveform.float().unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect").squeeze(1)
        frames = x.unfold(dimension=1, size=self.n_fft, step=self.hop)
        window = self.spectrogram.window
        if window.numel() != self.n_fft:
            left = (self.n_fft - window.numel()) // 2
            window = F.pad(window, (left, self.n_fft - window.numel() - left))
        spec = torch.fft.rfft(frames * window, dim=-1)
        power = spec.real.square() + spec.imag.square()
        mel = 10.0 * torch.log10(torch.clamp(power @ self.mel_scale.fb, min=1e-10))
        return (mel[:, :-1] - self.mel_mean) / torch.clamp(self.mel_std, min=1e-5)


class GlobalResponseNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, width))
        self.bias = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        normalized = magnitude / (torch.mean(magnitude, dim=-1, keepdim=True) + 1e-6)
        return self.weight * x * normalized + self.bias + x


class ConvNextLayer(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.depthwise_block = nn.Sequential(
            _Transpose(),
            nn.Conv1d(width, width, 7, padding=3, groups=width),
            _Transpose(),
        )
        self.pointwise_block = nn.Sequential(
            nn.LayerNorm(width, eps=eps),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            GlobalResponseNorm(4 * width),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise_block(self.depthwise_block(x))


class ConvNextBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, depth: int, eps: float):
        super().__init__()
        if cin != cout or stride > 1:
            self.resampling_layer = nn.Sequential(
                nn.LayerNorm(cin, eps=eps),
                _Transpose(),
                nn.Conv1d(cin, cout, 2, stride=stride),
                _Transpose(),
            )
        else:
            self.resampling_layer = nn.Sequential()
        self.convnext_layers = nn.ModuleList([ConvNextLayer(cout, eps) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resampling_layer(x)
        for layer in self.convnext_layers:
            x = layer(x)
        return x


def _apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    """Apply NeoX-style (half-split) RoPE to [B, H, T, D]."""
    t, dim = x.shape[2], x.shape[3]
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    pos = torch.arange(t, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos()[None, None, :, :].to(dtype=x.dtype)
    sin = freqs.sin()[None, None, :, :].to(dtype=x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class MertAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.base = float(config["rotary_embedding_base"])
        self.query_proj = nn.Linear(width, width)
        self.key_proj = nn.Linear(width, width)
        self.value_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, width = x.shape
        dim = width // self.heads
        q = self.query_proj(x).reshape(b, t, self.heads, dim).transpose(1, 2)
        k = self.key_proj(x).reshape(b, t, self.heads, dim).transpose(1, 2)
        v = self.value_proj(x).reshape(b, t, self.heads, dim).transpose(1, 2)
        q = _apply_rope(q, self.base)
        k = _apply_rope(k, self.base)
        # Tile queries in chunks of 512 so attention stays memory-bounded even on CPU fallback.
        if x.device.type == "cuda":
            parts = [F.scaled_dot_product_attention(q[:, :, i : i + 512], k, v) for i in range(0, t, 512)]
        else:
            parts = [F.scaled_dot_product_attention(q[:, :, i : i + 256], k, v) for i in range(0, t, 256)]
        y = torch.cat(parts, dim=2).transpose(1, 2).contiguous().reshape(b, t, width)
        return self.out_proj(y)


class MertFeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.w_1 = nn.Linear(int(config["hidden_size"]), int(config["intermediate_size"]))
        self.w_2 = nn.Linear(int(config["intermediate_size"]), int(config["hidden_size"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.gelu(self.w_1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        kernel = int(config["conv_depthwise_kernel_size"])
        eps = float(config["layer_norm_eps"])
        self.layer_norm = nn.LayerNorm(width, eps=eps)
        self.conv_block = nn.Sequential(
            _Transpose(),
            nn.Conv1d(width, 2 * width, 1, bias=False),
            nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2, groups=width, bias=False),
            nn.Sequential(_Transpose(), nn.LayerNorm(width, eps=eps), _Transpose()),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            _Transpose(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(self.layer_norm(x))


class ConformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        eps = float(config["layer_norm_eps"])
        self.ffn1_layer_norm = nn.LayerNorm(width, eps=eps)
        self.attn_layer_norm = nn.LayerNorm(width, eps=eps)
        self.ffn2_layer_norm = nn.LayerNorm(width, eps=eps)
        self.final_layer_norm = nn.LayerNorm(width, eps=eps)
        self.ffn1 = MertFeedForward(config)
        self.ffn2 = MertFeedForward(config)
        self.attn = MertAttention(config)
        self.conv_module = ConvolutionModule(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = x + self.attn(self.attn_layer_norm(x))
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class MERT2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_extractor = MelFrontend(config)
        channels = [int(config["num_mel_bins"]), *[int(c) for c in config["subsampling_channels"]]]
        self.subsampling_module = nn.ModuleList(
            [
                ConvNextBlock(
                    channels[i],
                    channels[i + 1],
                    (1, 2, 2)[i],
                    int(config["subsampling_depths"][i]),
                    float(config["subsampling_layer_norm_eps"]),
                )
                for i in range(3)
            ]
        )
        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(int(config["num_hidden_layers"]))])

    def forward(
        self,
        waveform: torch.Tensor,
        layer_weights: torch.Tensor | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> torch.Tensor:
        check_cancelled(cancelled)
        x = self.feature_extractor(waveform)
        for block in self.subsampling_module:
            x = block(x)
        mixed = None if layer_weights is None else x * layer_weights[0]
        for i, layer in enumerate(self.layers):
            check_cancelled(cancelled)
            x = layer(x)
            if mixed is not None:
                mixed = mixed + x * layer_weights[i + 1]
        return x if mixed is None else mixed


# ---------------------------------------------------------------------------
# Cached BART Decoder (official checkpoint layout)
# ---------------------------------------------------------------------------


class BartAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, width = x.shape
        dim = width // self.heads

        def split(val: torch.Tensor) -> torch.Tensor:
            return val.reshape(b, -1, self.heads, dim).transpose(1, 2)

        q = split(self.q_proj(x))
        if memory is not None and cache is not None:
            k, v = cache
        else:
            source = x if memory is None else memory
            k, v = split(self.k_proj(source)), split(self.v_proj(source))
            if cache is not None:
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.out_proj(attended.transpose(1, 2).contiguous().reshape(b, t, width))
        return out, (k, v)


class BartDecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        ff = int(config["intermediate_size"])
        self.self_attn = BartAttention(width, heads)
        self.encoder_attn = BartAttention(width, heads)
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.encoder_attn_layer_norm = nn.LayerNorm(width)
        self.final_layer_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, ff)
        self.fc2 = nn.Linear(ff, width)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, cache, mask: torch.Tensor | None):
        self_cache, cross_cache = cache if cache else (None, None)
        update, self_cache = self.self_attn(x, cache=self_cache, mask=mask)
        x = self.self_attn_layer_norm(x + update)
        update, cross_cache = self.encoder_attn(x, memory=memory, cache=cross_cache)
        x = self.encoder_attn_layer_norm(x + update)
        x = self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))
        return x, (self_cache, cross_cache)


class BartDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        self.embed_positions = nn.Embedding(int(config["max_output_seq_len"]) + 2, width)
        self.layernorm_embedding = nn.LayerNorm(width)
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(int(config["decoder_layers"]))])

    def forward(self, embedded: torch.Tensor, memory: torch.Tensor, cache=None):
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        seq_len = embedded.shape[1]
        positions = torch.arange(offset, offset + seq_len, device=embedded.device, dtype=torch.long)
        x = self.layernorm_embedding(embedded + self.embed_positions(positions + 2))
        mask = (
            None
            if (cache is not None and seq_len == 1)
            else (torch.arange(offset + seq_len, device=embedded.device)[None, :] <= positions[:, None])
        )
        updated = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, memory, None if cache is None else cache[i], mask)
            updated.append(state)
        return x, updated


# ---------------------------------------------------------------------------
# SheetSage2 Model & Loader
# ---------------------------------------------------------------------------


def resolve_models(
    model: str | Path = MODEL_REPO,
    base_model: str | Path = MERT_REPO,
    *,
    offline: bool = False,
    cache_dir: str | None = None,
) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    result: list[Path] = []
    for source, repo, revision in ((model, MODEL_REPO, MODEL_REVISION), (base_model, MERT_REPO, MERT_REVISION)):
        if Path(source).is_dir():
            result.append(Path(source))
        else:
            if str(source) != repo:
                raise ValueError(f"Use the pinned {repo} or a reviewed local directory")
            result.append(
                Path(
                    snapshot_download(
                        repo,
                        revision=revision,
                        local_files_only=offline,
                        cache_dir=cache_dir,
                        allow_patterns=["config.json", "model.safetensors"],
                    )
                )
            )
    return result[0], result[1]


class SheetSage2Torch(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder = MERT2(config["backbone_config"])
        self.layer_weight = nn.Parameter(torch.zeros(int(config["backbone_config"]["num_hidden_layers"]) + 1))
        self.encoder_projection = nn.Linear(int(config["backbone_config"]["hidden_size"]), int(config["hidden_size"]))
        self.token_embedding = nn.Embedding(int(config["vocab_size"]), int(config["hidden_size"]))
        self.decoder = BartDecoder(config)
        self.identity: dict = {}

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = MODEL_REPO,
        base_model: str | Path = MERT_REPO,
        *,
        device: str | torch.device = "auto",
        cancelled: Callable[[], bool] | None = None,
        **kwargs,
    ) -> "SheetSage2Torch":
        check_cancelled(cancelled)
        directory, parent = resolve_models(model, base_model, **kwargs)
        check_cancelled(cancelled)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("weights_format") != "adapter":
            raise ValueError("Expected the pinned SheetSage2 adapter checkpoint")
        parent_file = parent / "model.safetensors"
        parent_hash = sha256_file(parent_file)
        check_cancelled(cancelled)
        if parent_hash != config["base_model_sha256"]:
            raise ValueError("MERT2 checkpoint does not match SheetSage2's pinned parent")
        parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "sampling_rate",
            "hop_length",
            "n_fft",
            "win_length",
            "num_mel_bins",
            "conv_depthwise_kernel_size",
            "rotary_embedding_base",
            "subsampling_channels",
            "subsampling_depths",
            "layer_norm_eps",
            "subsampling_layer_norm_eps",
            "variant",
        ):
            if config["backbone_config"][key] != parent_config[key]:
                raise ValueError(f"MERT2 architecture mismatch: {key}")

        adapter = load_file(str(directory / "model.safetensors"))
        weights = load_file(str(parent_file))
        check_cancelled(cancelled)

        scale = float(config["lora_alpha"]) / float(config["lora_rank"])
        # Official loader merges LoRA on CPU in FP32 before inference.
        for key in list(adapter):
            check_cancelled(cancelled)
            if key.startswith("adapter.") and key.endswith("lora_A.weight"):
                target = key.removeprefix("adapter.").removesuffix("lora_A.weight") + "weight"
                bkey = key.replace("lora_A.weight", "lora_B.weight")
                weights[target] = weights[target] + (adapter[bkey] @ adapter[key]) * scale

        state_dict: dict[str, torch.Tensor] = {}
        for key, value in weights.items():
            state_dict["encoder." + key] = torch.from_numpy(np.array(value, copy=True))
        for key, value in adapter.items():
            if not key.startswith("adapter."):
                state_dict[key] = torch.from_numpy(np.array(value, copy=True))

        instance = cls(config)
        instance.load_state_dict(state_dict, strict=True)

        if device == "auto":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(device)
        if target_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        instance.to(device=target_device, dtype=torch.float32).eval()
        check_cancelled(cancelled)

        instance.identity = {
            "model": MODEL_REPO,
            "revision": MODEL_REVISION,
            "model_sha256": sha256_file(directory / "model.safetensors"),
            "config_sha256": sha256_file(directory / "config.json"),
            "base_model": MERT_REPO,
            "base_revision": MERT_REVISION,
            "base_sha256": parent_hash,
            "backend": "torch",
            "device": str(target_device),
            "dtype": "float32",
            "lora_merge": "numpy_cpu_fp32",
        }
        return instance

    @property
    def device(self) -> torch.device:
        return self.layer_weight.device

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray, cancelled: Callable[[], bool] | None = None) -> torch.Tensor:
        check_cancelled(cancelled)
        window = round(float(self.config["input_audio_length"]) * int(self.config["sampling_rate"]))
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim != 1 or not 1025 <= len(arr) <= window or not np.isfinite(arr).all():
            raise ValueError("Expected finite mono audio inside one SheetSage2 window")
        if len(arr) < window:
            arr = np.pad(arr, (0, window - len(arr)))
        wav_t = torch.from_numpy(arr).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        mixed = self.encoder(wav_t, torch.softmax(self.layer_weight, dim=0), cancelled)
        return self.encoder_projection(mixed)

    @torch.inference_mode()
    def decode(self, memory: torch.Tensor, ids: list[int], cache=None):
        tok_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden, cache = self.decoder(self.token_embedding(tok_t), memory, cache)
        logits = F.linear(hidden, self.token_embedding.weight)
        return logits, cache


# ---------------------------------------------------------------------------
# Full-Song Transcription Pipeline
# ---------------------------------------------------------------------------


def decode_audio(command: list[str], cancelled=None, timeout: int = 600) -> np.ndarray:
    """Drain FFmpeg stdout/stderr while polling cancellation, always reaping FFmpeg."""
    check_cancelled(cancelled)
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                check_cancelled(cancelled)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            check_cancelled(cancelled)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            return np.frombuffer(stdout, dtype="<f4").copy()
        except BaseException:
            process.kill()
            process.communicate()
            raise


@torch.inference_mode()
def generate_tokens(
    model: SheetSage2Torch,
    tokenizer: SheetSage2Tokenizer,
    waveform: np.ndarray,
    prompts: list[str],
    prefix: list[int] | None = None,
    stop_time: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, bool]:
    check_cancelled(cancelled)
    prefix = list(prefix) if prefix is not None else tokenizer.prompt_prefix(prompts)
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]
    if len(prefix) >= model.config["max_output_seq_len"]:
        raise ValueError("Overlap prefix fills the decoder context")

    state = PromptGrammarState(tokenizer)
    for token in prefix[prefix.index(tokenizer.out_token) + 1 :]:
        state.update(token)

    memory = model.encode(waveform, cancelled)
    cache, ids, tokens = None, prefix, list(prefix)
    ended = False
    while len(tokens) < model.config["max_output_seq_len"]:
        check_cancelled(cancelled)
        logits, cache = model.decode(memory, ids, cache)
        step_logits = logits[0, -1].float()
        allowed_np = state.allowed(None)
        allowed = torch.from_numpy(np.asarray(allowed_np, dtype=np.bool_)).to(device=step_logits.device)
        masked = step_logits.masked_fill(~allowed, float("-inf"))
        token = int(torch.argmax(masked).item())
        tokens.append(token)
        ended = state.update(token)
        if not ended and stop_time is not None and tokenizer.time_token_start <= token < tokenizer.time_token_end:
            if tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_time:
                tokens.append(tokenizer.eos_token)
                ended = True
        if ended:
            break
        ids = [token]
        if progress is not None and len(tokens) % 100 == 0:
            progress({"stage": "decoding", "tokens": len(tokens)})

    truncated = not ended
    if tokens[-1] != tokenizer.eos_token:
        tokens.append(tokenizer.eos_token)
    return np.asarray(tokens, dtype=np.int64), truncated


def transcribe(
    audio: str | Path,
    output: str | Path,
    *,
    model: SheetSage2Torch | None = None,
    model_path: str | Path = MODEL_REPO,
    base_model: str | Path = MERT_REPO,
    offline: bool = False,
    cache_dir: str | None = None,
    device: str | torch.device = "auto",
    task: str = "melody-full",
    preset: str = "default",
    max_seconds: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    source, output = Path(audio), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Transcription needs a fresh output directory")
    if task not in {"full", "melody-full", "melody-vocal"} or preset not in {"default", "paper"}:
        raise ValueError("Invalid transcription task or window preset")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive; omit to process the complete recording")

    check_cancelled(cancelled)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source.resolve()), "-vn"]
    if max_seconds is not None:
        command += ["-t", str(max_seconds)]
    command += ["-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1"]
    waveform = decode_audio(command, cancelled)
    if len(waveform) < 1025 or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain at least 1025 finite samples at 24 kHz")

    start = time.perf_counter()
    check_cancelled(cancelled)
    if model is None:
        model = SheetSage2Torch.from_pretrained(
            model_path,
            base_model,
            device=device,
            offline=offline,
            cache_dir=cache_dir,
            cancelled=cancelled,
        )
    check_cancelled(cancelled)
    cfg = model.config
    tokenizer = SheetSage2Tokenizer(
        cfg["input_audio_length"],
        cfg["time_hz"],
        cfg["tokenizer_schema_version"],
        expected_fingerprint=cfg["tokenizer_fingerprint"],
    )
    prompts = ["timestamp", "downbeat_meter", "structure", "key"]
    prompts += ["chord_full", "melody_full"] if task == "full" else ["melody_vocal" if task == "melody-vocal" else "melody_full"]
    prompts = tokenizer.normalize_prompts(prompts)

    duration, length = len(waveform) / 24000.0, cfg["input_audio_length"]
    windows = sliding_window_plan(
        duration,
        length,
        200 if preset == "default" else 100,
        100 if preset == "default" else 0,
    )
    stitched, records, warnings = [], [], []
    if preset == "paper":
        warnings.append("Paper window/export preset with FFmpeg resampling; not a reproduction of the torchaudio paper frontend")
    output.mkdir(parents=True, exist_ok=True)

    for index, window in enumerate(windows):
        check_cancelled(cancelled)
        if progress:
            progress({"stage": "encoding", "window": index + 1, "windows": len(windows)})
        prefix, base = None, 0
        if index:
            _, prefix, base = build_overlap_prefix_tokens(stitched, tokenizer, prompts, window["start"], window["prefix_end"])
            limit = cfg["max_output_seq_len"] - (16 if preset == "paper" else 128)
            if prefix is not None and len(prefix) >= limit:
                if preset == "paper":
                    prefix = None
                    warnings.append("Overlap prefix exceeded the paper context")
                else:
                    raise ValueError("Overlap prefix fills context; transcription cannot continue faithfully")
        offset = round(window["start"] * 24000)
        segment = waveform[offset : offset + round(length * 24000)]
        stop_time = (
            None
            if preset == "paper"
            else (window["generation_stop"] if window["generation_stop"] is not None else min(duration - window["start"], length))
        )
        tokens, truncated = generate_tokens(model, tokenizer, segment, prompts, prefix, stop_time, cancelled, progress)
        decoded, warning = decode_generated_tokens(tokenizer, tokens, source.stem, index)
        if warning:
            warnings.append(warning["error"])
        stitched.extend(
            stitched_window_events(
                decoded,
                event_time_map(decoded, length),
                window["start"],
                window["accept_start"],
                window["accept_end"],
                duration,
                index,
                base or 0,
            )
        )
        if preset == "paper":
            stitched.sort(
                key=lambda e: (
                    float(e.get("time", 0)),
                    int(e.get("window_index", 0)),
                    int(e.get("source_subbeat", e["subbeat"])),
                )
            )
        records.append({**window, "tokens": tokens.tolist(), "truncated": truncated})
        check_cancelled(cancelled)
        write_json(output / "windows.json", records)

    if preset != "paper":
        stitched.sort(key=lambda e: (e["time"], e["global_subbeat"]))
    decoded = dict(schema_version=tokenizer.schema_version, prompts=list(prompts), events=stitched, has_eos=True)
    check_cancelled(cancelled)
    exported = export_result(decoded, tokenizer, output, duration, paper=preset == "paper", melody_only=task != "full")
    payload = exported.pop("payload")
    result = {
        **exported,
        "backend": "torch",
        "dtype": "float32",
        "model": model.identity,
        "source_audio_sha256": sha256_file(source),
        "duration_seconds": duration,
        "task": task,
        "preset": preset,
        "audio_decode": "ffmpeg_float32_mono_24000",
        "warnings": warnings,
        "truncated": any(r["truncated"] for r in records),
        "elapsed_seconds": time.perf_counter() - start,
    }
    result["status"] = "complete" if payload.get("abc") and not result.get("abc_error") else "failed"
    check_cancelled(cancelled)
    write_json(output / "result.json", result)
    return {**result, **payload}
