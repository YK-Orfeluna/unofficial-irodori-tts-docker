from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import torch
from safetensors import safe_open

from irodori_tts.inference_runtime import (
    RuntimeKey,
    clear_cached_runtime,
    download_hf_checkpoint,
    get_cached_runtime,
    list_available_runtime_precisions,
)
from irodori_tts.quantization import normalize_quantization_type, parse_quantization_metadata
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"

# モデル切替で旧モデルと新モデルが8GB VRAMを同時に占有することを防ぐ。
RUNTIME_LOCK = threading.RLock()
_runtime_key: RuntimeKey | None = None


def load_runtime(key: RuntimeKey):
    global _runtime_key
    with RUNTIME_LOCK:
        if key != _runtime_key:
            clear_cached_runtime()
            _runtime_key = None
        try:
            runtime, reloaded = get_cached_runtime(key)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        _runtime_key = key
        return runtime, reloaded


def unload_runtime() -> None:
    global _runtime_key
    with RUNTIME_LOCK:
        clear_cached_runtime()
        _runtime_key = None


def runtime_precision_choices(device: str) -> list[str]:
    choices = list_available_runtime_precisions(device)
    if torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        return [value for value in choices if value != "bf16"]
    return choices


def _validate_quantization(quantization_type: str, device: str) -> None:
    required = {
        "int4_weight_only": (8, 0),
        "float8_weight_only": (8, 9),
        "float8_dynamic_activation_float8_weight": (8, 9),
    }.get(quantization_type)
    if required is None:
        return
    if torch.device(device).type != "cuda" or torch.cuda.get_device_capability(device) < required:
        raise ValueError(
            f"{quantization_type} requires an NVIDIA GPU with compute capability "
            f"{required[0]}.{required[1]} or newer."
        )


def resolve_precision_from_env(
    env_var_name: str,
    *,
    device: str,
    fallback: str,
    log_prefix: str = "startup",
) -> str:
    """env var に指定された precision を読み取り、device 上で有効か検証する。

    未設定/空文字なら fallback を返す。不正値ならエラー出力後 SystemExit。
    silent fallback を避け、設定ミスを起動失敗で気づけるようにする。
    """
    raw = os.environ.get(env_var_name)
    if raw is None or raw.strip() == "":
        print(
            f"[{log_prefix}] {env_var_name} not set; using fallback={fallback!r} "
            f"(device={device!r})",
            flush=True,
        )
        return fallback
    value = raw.strip()
    allowed = runtime_precision_choices(device)
    if value not in allowed:
        print(
            f"[{log_prefix}] invalid {env_var_name}={value!r} for device={device!r}. "
            f"allowed values: {allowed}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    print(
        f"[{log_prefix}] {env_var_name}={value!r} (device={device!r})",
        flush=True,
    )
    return value


def resolve_checkpoint_path(
    raw_checkpoint: str,
    *,
    log_prefix: str,
    reject_speaker_inversion: bool = False,
) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    if reject_speaker_inversion and is_speaker_inversion_safetensors_path(checkpoint):
        raise ValueError("Speaker embedding files cannot be used as model checkpoints.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = download_hf_checkpoint(checkpoint)
    print(f"[{log_prefix}] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def build_runtime_key(
    *,
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    log_prefix: str,
    reject_speaker_inversion: bool = False,
) -> RuntimeKey:
    if model_precision not in runtime_precision_choices(model_device):
        raise ValueError(f"Unsupported model precision {model_precision!r} on {model_device}.")
    if codec_precision not in runtime_precision_choices(codec_device):
        raise ValueError(f"Unsupported codec precision {codec_precision!r} on {codec_device}.")
    if "-Quantized/" in checkpoint:
        _validate_quantization(normalize_quantization_type(checkpoint.rsplit("/", 1)[-1]), model_device)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint,
        log_prefix=log_prefix,
        reject_speaker_inversion=reject_speaker_inversion,
    )
    if Path(checkpoint_path).suffix.lower() == ".safetensors":
        with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            quantization = parse_quantization_metadata(handle.metadata() or {})
        if quantization is not None:
            _validate_quantization(quantization["quantization_type"], model_device)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=CODEC_REPO,
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )
