from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from irodori_tts.inference_runtime import (
    SamplingRequest,
    SamplingResult,
    get_cached_runtime,
    save_wav,
)

from .runtime_helpers import build_runtime_key
from .text_splitter import split_text_for_tts

# シリアル化はモデル単一GPU・既存Gradio queue(concurrency_limit=1)に揃える。
# threading.Lock を使うことで Gradio (同期ハンドラ)・FastAPI (sync または run_in_threadpool 経由) どちらからでも同じ排他を共有できる。
_SYNTH_LOCK = threading.Lock()

_OUTPUT_DIR_ENV = "IRODORI_API_OUTPUT_DIR"
_DEFAULT_OUTPUT_DIR = Path("/app/api_outputs")
MAX_CANDIDATES = 32


def get_output_dir() -> Path:
    raw = os.environ.get(_OUTPUT_DIR_ENV)
    out = Path(raw) if raw else _DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class CandidateOutput:
    index: int
    path: str  # コンテナ内絶対パス
    url: str  # /files/<basename> 形式の相対URL


@dataclass
class SynthesisOutput:
    service: str
    sample_rate: int
    used_seed: int
    candidates: list[CandidateOutput]
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    messages: list[str]


# 長文 auto_split のデフォルト値。READMEと一致させること。
DEFAULT_MAX_CHARS_PER_CHUNK = 200
DEFAULT_CHUNK_SILENCE_MS = 150


@dataclass
class RefSynthesisParams:
    text: str
    ref_wav: str  # ローカルファイルパス(API ハンドラが multipart を一時保存したパス)
    num_steps: int = 40
    num_candidates: int = 1
    seed: int | None = None
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    duration_scale: float = 1.0
    checkpoint: str | None = None
    model_device: str | None = None
    model_precision: str | None = None
    codec_device: str | None = None
    codec_precision: str | None = None
    auto_split: bool = False
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK
    chunk_silence_ms: int = DEFAULT_CHUNK_SILENCE_MS


@dataclass
class VoiceDesignSynthesisParams:
    text: str
    caption: str = ""
    ref_wav: str | None = None
    num_steps: int = 40
    num_candidates: int = 1
    seed: int | None = None
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 4.0
    cfg_scale_speaker: float = 5.0
    duration_scale: float = 1.0
    checkpoint: str | None = None
    model_device: str | None = None
    model_precision: str | None = None
    codec_device: str | None = None
    codec_precision: str | None = None
    auto_split: bool = False
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK
    chunk_silence_ms: int = DEFAULT_CHUNK_SILENCE_MS


def _validate_common(text: str, num_candidates: int) -> None:
    if str(text).strip() == "":
        raise ValueError("text is required.")
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1.")
    if num_candidates > MAX_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_CANDIDATES}.")


def _validate_split_params(
    *, auto_split: bool, num_candidates: int, max_chars: int, silence_ms: int
) -> None:
    if not auto_split:
        return
    if num_candidates != 1:
        # 分割時に複数候補を出すと chunk × candidate の組み合わせが意味不明になるため明示拒否。
        raise ValueError("auto_split requires num_candidates == 1.")
    if max_chars < 1:
        raise ValueError("max_chars_per_chunk must be >= 1.")
    if silence_ms < 0:
        raise ValueError("chunk_silence_ms must be >= 0.")


def _silence_tensor(reference: torch.Tensor, sample_rate: int, ms: int) -> torch.Tensor:
    n_samples = max(0, int(sample_rate * ms / 1000))
    if n_samples == 0:
        return reference.new_zeros((*reference.shape[:-1], 0))
    return reference.new_zeros((*reference.shape[:-1], n_samples))


def _concat_chunks(audios: list[torch.Tensor], sample_rate: int, silence_ms: int) -> torch.Tensor:
    if len(audios) == 1:
        return audios[0]
    silence = _silence_tensor(audios[0], sample_rate, silence_ms)
    pieces: list[torch.Tensor] = []
    for i, audio in enumerate(audios):
        if i > 0 and silence.shape[-1] > 0:
            pieces.append(silence)
        pieces.append(audio)
    return torch.cat(pieces, dim=-1)


def run_split_synthesis(  # 公開: gradio_app からも再利用するため
    *,
    chunks: list[str],
    build_request: Callable[[str, int | None], SamplingRequest],
    initial_seed: int | None,
    runtime,
    log_fn: Callable[[str], None],
    silence_ms: int,
) -> SamplingResult:
    """chunk 列を順次合成し、無音を挟んで連結した SamplingResult を返す。

    - 全 chunk で同一 seed を使う(声質の一貫性)。
    - 各 chunk は num_candidates=1 (build_request 側で保証する想定)。
    - stage_timings は chunk_i/<stage_name> プレフィクスで集約。
    """
    audios: list[torch.Tensor] = []
    timings: list[tuple[str, float]] = []
    messages: list[str] = []
    total_decode = 0.0
    seed_for_call = initial_seed
    used_seed = 0
    sample_rate = 0

    for i, chunk_text in enumerate(chunks, start=1):
        log_fn(f"[split] chunk {i}/{len(chunks)} ({len(chunk_text)} chars)")
        result = runtime.synthesize(build_request(chunk_text, seed_for_call), log_fn=log_fn)
        if not result.audios:
            raise RuntimeError(f"chunk {i} produced no audio.")
        audios.append(result.audios[0])
        timings.extend((f"chunk_{i}/{name}", sec) for name, sec in result.stage_timings)
        messages.extend(f"chunk_{i}: {m}" for m in result.messages)
        total_decode += result.total_to_decode
        if seed_for_call is None:
            # 後続 chunk に同じ seed を使い回す。
            seed_for_call = result.used_seed
        used_seed = result.used_seed
        sample_rate = result.sample_rate

    concatenated = _concat_chunks(audios, sample_rate, silence_ms)
    messages.insert(0, f"split into {len(chunks)} chunks (silence={silence_ms}ms)")
    return SamplingResult(
        audio=concatenated,
        audios=[concatenated],
        sample_rate=sample_rate,
        stage_timings=timings,
        total_to_decode=total_decode,
        used_seed=used_seed,
        messages=messages,
    )


def _generate_stem() -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return f"sample_{stamp}_{uuid.uuid4().hex[:8]}"


def _save_candidates(
    audios: list,
    sample_rate: int,
    stem: str,
) -> list[CandidateOutput]:
    out_dir = get_output_dir()
    results: list[CandidateOutput] = []
    for i, audio in enumerate(audios, start=1):
        name = f"{stem}_{i:03d}.wav"
        path = save_wav(out_dir / name, audio.float(), sample_rate)
        results.append(
            CandidateOutput(index=i, path=str(path), url=f"/files/{name}")
        )
    return results


def synthesize_ref(
    params: RefSynthesisParams,
    *,
    default_checkpoint: str,
    default_model_device: str,
    default_model_precision: str,
    default_codec_device: str,
    default_codec_precision: str,
    log_prefix: str = "api-ref",
) -> SynthesisOutput:
    _validate_common(params.text, params.num_candidates)
    if not params.ref_wav:
        raise ValueError("ref_wav is required for the reference-audio endpoint.")
    _validate_split_params(
        auto_split=params.auto_split,
        num_candidates=params.num_candidates,
        max_chars=params.max_chars_per_chunk,
        silence_ms=params.chunk_silence_ms,
    )

    runtime_key = build_runtime_key(
        checkpoint=params.checkpoint or default_checkpoint,
        model_device=params.model_device or default_model_device,
        model_precision=params.model_precision or default_model_precision,
        codec_device=params.codec_device or default_codec_device,
        codec_precision=params.codec_precision or default_codec_precision,
        log_prefix=log_prefix,
        reject_speaker_inversion=False,
    )

    log_fn = lambda msg: print(msg, flush=True)  # noqa: E731

    def build_request(text: str, seed: int | None) -> SamplingRequest:
        return SamplingRequest(
            text=text,
            ref_wav=params.ref_wav,
            ref_latent=None,
            ref_embed=None,
            no_ref=False,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1 if params.auto_split else int(params.num_candidates),
            decode_mode="sequential",
            seconds=None,
            duration_scale=float(params.duration_scale),
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=int(params.num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode="independent",
            cfg_scale_text=float(params.cfg_scale_text),
            cfg_scale_speaker=float(params.cfg_scale_speaker),
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            t_schedule_mode="linear",
            sway_coeff=-1.0,
            trim_tail=True,
            lora_adapter=None,
        )

    with _SYNTH_LOCK:
        runtime, _ = get_cached_runtime(runtime_key)
        if params.auto_split:
            chunks = split_text_for_tts(params.text, max_chars=int(params.max_chars_per_chunk))
            if not chunks:
                raise ValueError("text is required.")
            result = run_split_synthesis(
                chunks=chunks,
                build_request=build_request,
                initial_seed=params.seed,
                runtime=runtime,
                log_fn=log_fn,
                silence_ms=int(params.chunk_silence_ms),
            )
        else:
            # auto_split=False では改行をそのまま渡すと合成音が乱れるため空白に置換する
            sanitized_text = str(params.text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            result = runtime.synthesize(build_request(sanitized_text, params.seed), log_fn=log_fn)

    candidates = _save_candidates(result.audios, result.sample_rate, _generate_stem())
    return SynthesisOutput(
        service="irodori-tts-500m",
        sample_rate=result.sample_rate,
        used_seed=result.used_seed,
        candidates=candidates,
        stage_timings=list(result.stage_timings),
        total_to_decode=result.total_to_decode,
        messages=list(result.messages),
    )


def synthesize_voicedesign(
    params: VoiceDesignSynthesisParams,
    *,
    default_checkpoint: str,
    default_model_device: str,
    default_model_precision: str,
    default_codec_device: str,
    default_codec_precision: str,
    log_prefix: str = "api-voicedesign",
) -> SynthesisOutput:
    _validate_common(params.text, params.num_candidates)
    _validate_split_params(
        auto_split=params.auto_split,
        num_candidates=params.num_candidates,
        max_chars=params.max_chars_per_chunk,
        silence_ms=params.chunk_silence_ms,
    )

    runtime_key = build_runtime_key(
        checkpoint=params.checkpoint or default_checkpoint,
        model_device=params.model_device or default_model_device,
        model_precision=params.model_precision or default_model_precision,
        codec_device=params.codec_device or default_codec_device,
        codec_precision=params.codec_precision or default_codec_precision,
        log_prefix=log_prefix,
        reject_speaker_inversion=True,
    )

    caption_value = (params.caption or "").strip()
    ref_wav_path = params.ref_wav if (params.ref_wav and str(params.ref_wav).strip()) else None
    log_fn = lambda msg: print(msg, flush=True)  # noqa: E731

    with _SYNTH_LOCK:
        runtime, _ = get_cached_runtime(runtime_key)
        if not runtime.model_cfg.use_caption_condition:
            raise ValueError(
                "Loaded checkpoint does not enable caption conditioning. Use the 500M reference endpoint."
            )

        effective_no_ref = (
            ref_wav_path is None or not runtime.model_cfg.use_speaker_condition_resolved
        )
        ref_wav_for_request = None if effective_no_ref else ref_wav_path

        def build_request(text: str, seed: int | None) -> SamplingRequest:
            return SamplingRequest(
                text=text,
                caption=caption_value or None,
                ref_wav=ref_wav_for_request,
                ref_latent=None,
                no_ref=effective_no_ref,
                ref_normalize_db=-16.0,
                ref_ensure_max=True,
                num_candidates=1 if params.auto_split else int(params.num_candidates),
                decode_mode="sequential",
                seconds=None,
                duration_scale=float(params.duration_scale),
                max_ref_seconds=30.0,
                max_text_len=None,
                max_caption_len=None,
                num_steps=int(params.num_steps),
                seed=None if seed is None else int(seed),
                cfg_guidance_mode="independent",
                cfg_scale_text=float(params.cfg_scale_text),
                cfg_scale_caption=float(params.cfg_scale_caption),
                cfg_scale_speaker=0.0 if effective_no_ref else float(params.cfg_scale_speaker),
                cfg_scale=None,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                truncation_factor=None,
                rescale_k=None,
                rescale_sigma=None,
                context_kv_cache=True,
                speaker_kv_scale=None,
                speaker_kv_min_t=None,
                speaker_kv_max_layers=None,
                t_schedule_mode="linear",
                sway_coeff=-1.0,
                trim_tail=True,
                lora_adapter=None,
            )

        if params.auto_split:
            chunks = split_text_for_tts(params.text, max_chars=int(params.max_chars_per_chunk))
            if not chunks:
                raise ValueError("text is required.")
            result = run_split_synthesis(
                chunks=chunks,
                build_request=build_request,
                initial_seed=params.seed,
                runtime=runtime,
                log_fn=log_fn,
                silence_ms=int(params.chunk_silence_ms),
            )
        else:
            # auto_split=False では改行をそのまま渡すと合成音が乱れるため空白に置換する
            sanitized_text = str(params.text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            result = runtime.synthesize(build_request(sanitized_text, params.seed), log_fn=log_fn)

    candidates = _save_candidates(result.audios, result.sample_rate, _generate_stem())
    return SynthesisOutput(
        service="irodori-tts-voicedesign",
        sample_rate=result.sample_rate,
        used_seed=result.used_seed,
        candidates=candidates,
        stage_timings=list(result.stage_timings),
        total_to_decode=result.total_to_decode,
        messages=list(result.messages),
    )
