from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Literal

import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from irodori_tts.inference_runtime import get_cached_runtime

from .inference_facade import (
    DEFAULT_CHUNK_SILENCE_MS,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    RefSynthesisParams,
    SynthesisOutput,
    VoiceDesignSynthesisParams,
    get_output_dir,
    synthesize_ref,
    synthesize_voicedesign,
)
from .runtime_helpers import build_runtime_key

_FILENAME_RE = re.compile(r"^sample_[A-Za-z0-9_-]+\.wav$")

REFERENCE_AUDIO_ROOT = Path("/app/reference_audio")


def _validate_reference_path(raw: str) -> str:
    """Restrict ref_wav_path to files under /app/reference_audio (no traversal)."""
    if not raw or not str(raw).strip():
        raise HTTPException(status_code=400, detail="ref_wav_path is empty.")
    candidate = Path(str(raw).strip())
    if not candidate.is_absolute():
        candidate = REFERENCE_AUDIO_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"invalid ref_wav_path: {exc}") from exc
    root = REFERENCE_AUDIO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(
            status_code=400,
            detail=f"ref_wav_path must be under {REFERENCE_AUDIO_ROOT}.",
        )
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"ref_wav_path not found: {resolved}")
    return str(resolved)


def _output_dict(output: SynthesisOutput) -> dict:
    return {
        "service": output.service,
        "sample_rate": output.sample_rate,
        "used_seed": output.used_seed,
        "candidates": [asdict(c) for c in output.candidates],
        "stage_timings": [[name, sec] for name, sec in output.stage_timings],
        "total_to_decode": output.total_to_decode,
        "messages": output.messages,
    }


def _wrap_synthesis_errors(fn: Callable[[], SynthesisOutput]) -> JSONResponse:
    try:
        out = fn()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"file not found: {exc}") from exc
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=507, detail=f"CUDA OOM: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        # MVP: 詳細な区別はせず 500 として返却。詳細はサーバログで追跡。
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse(content=_output_dict(out))


def attach_api(
    app: FastAPI,
    *,
    endpoint_kind: Literal["ref", "voicedesign"],
    default_checkpoint: str,
    default_model_device: str,
    default_model_precision: str,
    default_codec_device: str,
    default_codec_precision: str,
) -> None:
    service_name = (
        "irodori-tts-500m" if endpoint_kind == "ref" else "irodori-tts-voicedesign"
    )
    log_prefix = "api-ref" if endpoint_kind == "ref" else "api-voicedesign"

    @app.on_event("startup")
    def _preload_runtime() -> None:
        # アイドル時の VRAM 未確保問題対策: lazy load を避け、起動直後に確定の precision でモデルを VRAM に載せる。
        print(
            f"[{log_prefix}] preloading runtime at startup "
            f"(model={default_model_device}/{default_model_precision}, "
            f"codec={default_codec_device}/{default_codec_precision})",
            flush=True,
        )
        runtime_key = build_runtime_key(
            checkpoint=default_checkpoint,
            model_device=default_model_device,
            model_precision=default_model_precision,
            codec_device=default_codec_device,
            codec_precision=default_codec_precision,
            log_prefix=log_prefix,
            reject_speaker_inversion=(endpoint_kind == "voicedesign"),
        )
        get_cached_runtime(runtime_key)
        print(f"[{log_prefix}] preload complete", flush=True)

    @app.get("/api/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": service_name,
            "default_checkpoint": default_checkpoint,
        }

    if endpoint_kind == "ref":

        @app.post("/api/v1/synthesize")
        def synthesize_ref_endpoint(
            text: str = Form(...),
            ref_wav_path: str = Form(...),
            num_steps: int = Form(40),
            num_candidates: int = Form(1),
            seed: int | None = Form(None),
            cfg_scale_text: float = Form(3.0),
            cfg_scale_speaker: float = Form(5.0),
            duration_scale: float = Form(1.0),
            checkpoint: str | None = Form(None),
            auto_split: bool = Form(False),
            max_chars_per_chunk: int = Form(DEFAULT_MAX_CHARS_PER_CHUNK),
            chunk_silence_ms: int = Form(DEFAULT_CHUNK_SILENCE_MS),
        ) -> JSONResponse:
            resolved_ref = _validate_reference_path(ref_wav_path)
            return _wrap_synthesis_errors(
                lambda: synthesize_ref(
                    RefSynthesisParams(
                        text=text,
                        ref_wav=resolved_ref,
                        num_steps=num_steps,
                        num_candidates=num_candidates,
                        seed=seed,
                        cfg_scale_text=cfg_scale_text,
                        cfg_scale_speaker=cfg_scale_speaker,
                        duration_scale=duration_scale,
                        checkpoint=checkpoint,
                        auto_split=auto_split,
                        max_chars_per_chunk=max_chars_per_chunk,
                        chunk_silence_ms=chunk_silence_ms,
                    ),
                    default_checkpoint=default_checkpoint,
                    default_model_device=default_model_device,
                    default_model_precision=default_model_precision,
                    default_codec_device=default_codec_device,
                    default_codec_precision=default_codec_precision,
                    log_prefix=log_prefix,
                )
            )

    else:

        @app.post("/api/v1/synthesize")
        def synthesize_voicedesign_endpoint(
            text: str = Form(...),
            caption: str = Form(""),
            ref_wav_path: str | None = Form(None),
            num_steps: int = Form(40),
            num_candidates: int = Form(1),
            seed: int | None = Form(None),
            cfg_scale_text: float = Form(3.0),
            cfg_scale_caption: float = Form(4.0),
            cfg_scale_speaker: float = Form(5.0),
            duration_scale: float = Form(1.0),
            checkpoint: str | None = Form(None),
            auto_split: bool = Form(False),
            max_chars_per_chunk: int = Form(DEFAULT_MAX_CHARS_PER_CHUNK),
            chunk_silence_ms: int = Form(DEFAULT_CHUNK_SILENCE_MS),
        ) -> JSONResponse:
            resolved_ref = (
                _validate_reference_path(ref_wav_path)
                if ref_wav_path and ref_wav_path.strip()
                else None
            )
            return _wrap_synthesis_errors(
                lambda: synthesize_voicedesign(
                    VoiceDesignSynthesisParams(
                        text=text,
                        caption=caption,
                        ref_wav=resolved_ref,
                        num_steps=num_steps,
                        num_candidates=num_candidates,
                        seed=seed,
                        cfg_scale_text=cfg_scale_text,
                        cfg_scale_caption=cfg_scale_caption,
                        cfg_scale_speaker=cfg_scale_speaker,
                        duration_scale=duration_scale,
                        checkpoint=checkpoint,
                        auto_split=auto_split,
                        max_chars_per_chunk=max_chars_per_chunk,
                        chunk_silence_ms=chunk_silence_ms,
                    ),
                    default_checkpoint=default_checkpoint,
                    default_model_device=default_model_device,
                    default_model_precision=default_model_precision,
                    default_codec_device=default_codec_device,
                    default_codec_precision=default_codec_precision,
                    log_prefix=log_prefix,
                )
            )

    @app.get("/files/{name}")
    def serve_file(name: str) -> FileResponse:
        if not _FILENAME_RE.match(name):
            raise HTTPException(status_code=404, detail="not found")
        out_dir = get_output_dir().resolve()
        candidate = (out_dir / name).resolve()
        if candidate.parent != out_dir or not candidate.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path=str(candidate), media_type="audio/wav", filename=name)
