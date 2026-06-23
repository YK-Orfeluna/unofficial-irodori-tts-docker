from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import RuntimeKey, list_available_runtime_precisions
from irodori_tts.speaker_inversion import is_speaker_inversion_safetensors_path

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


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
    allowed = list_available_runtime_precisions(device)
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

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
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
    checkpoint_path = resolve_checkpoint_path(
        checkpoint,
        log_prefix=log_prefix,
        reject_speaker_inversion=reject_speaker_inversion,
    )
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
