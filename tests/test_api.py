from __future__ import annotations

from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from irodori_tts.inference_runtime import SamplingResult
from irodori_tts_api import app as api
from irodori_tts_api import inference_facade as facade


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    references = tmp_path / "references"
    references.mkdir()
    for name in ("first.wav", "second.wav"):
        (references / name).touch()
    monkeypatch.setattr(api, "REFERENCE_AUDIO_ROOT", references)
    monkeypatch.setenv("IRODORI_API_OUTPUT_DIR", str(tmp_path / "outputs"))
    requests = []

    def generate(request, **kwargs):
        requests.append(request)
        audio = torch.ones(1, 480) * 0.1
        return SamplingResult(
            audio, [audio] * request.num_candidates, 48000,
            [("sample_rf", 0.01)], 0.02, request.seed or 677, [],
        )

    runtime = SimpleNamespace(
        model_cfg=SimpleNamespace(use_caption_condition=True, use_speaker_condition_resolved=True),
        synthesize=generate,
    )
    monkeypatch.setattr(api, "build_runtime_key", lambda **kwargs: None)
    monkeypatch.setattr(api, "load_runtime", lambda key: (runtime, False))
    monkeypatch.setattr(facade, "build_runtime_key", lambda **kwargs: None)
    monkeypatch.setattr(facade, "load_runtime", lambda key: (runtime, False))
    app = FastAPI()
    api.attach_api(
        app, default_checkpoint="test-checkpoint", default_model_device="cpu",
        default_model_precision="fp32", default_codec_device="cpu", default_codec_precision="fp32",
    )
    with TestClient(app) as client:
        yield client, requests, runtime


def test_legacy_single_reference_request_and_response(endpoint):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", data={
        "text": "こんにちは。\r\n", "ref_wav_path": "first.wav", "seed": "677",
        "num_steps": "40", "cfg_scale_text": "3", "cfg_scale_speaker": "5",
    })
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "service", "sample_rate", "used_seed", "candidates", "stage_timings",
        "total_to_decode", "messages",
    }
    assert payload["service"] == "irodori-tts"
    assert payload["used_seed"] == 677
    assert set(payload["candidates"][0]) == {"index", "path", "url"}
    assert requests[0].ref_wavs[0].endswith("first.wav")
    assert requests[0].text == "こんにちは。 "
    assert requests[0].num_steps == 40
    assert requests[0].max_ref_seconds is None
    wav = client.get(payload["candidates"][0]["url"])
    assert wav.status_code == 200
    assert wav.content.startswith(b"RIFF")
    assert sf.info(payload["candidates"][0]["path"]).samplerate == 48000


def test_multiple_references_preserve_order(endpoint):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", files=[
        ("text", (None, "こんにちは。")),
        ("ref_wav_paths", (None, "second.wav")),
        ("ref_wav_paths", (None, "first.wav")),
        ("caption", (None, "落ち着いた声")),
    ])
    assert response.status_code == 200
    assert [path.rsplit("/", 1)[-1] for path in requests[0].ref_wavs] == ["second.wav", "first.wav"]
    assert requests[0].caption == "落ち着いた声"
    assert requests[0].num_steps is None


@pytest.mark.parametrize("caption", ["", "明るい声"])
def test_no_reference(endpoint, caption):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", data={"text": "こんにちは。", "caption": caption})
    assert response.status_code == 200
    assert requests[0].no_ref
    assert requests[0].ref_wavs is None
    assert requests[0].cfg_scale_speaker == 0


@pytest.mark.parametrize("fields", [
    {"ref_wav_path": "first.wav", "ref_wav_paths": "second.wav"},
    {"ref_wav_path": "../escape.wav"},
    {"ref_wav_paths": "../escape.wav"},
    {"ref_wav_path": "missing.wav"},
    {"ref_wav_path": " "},
    {"num_steps": "0"},
    {"auto_split": "true", "num_candidates": "2"},
])
def test_invalid_requests_return_400_without_sampling(endpoint, fields):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", data={"text": "こんにちは。", **fields})
    assert response.status_code == 400
    assert set(response.json()) == {"detail"}
    assert requests == []


def test_caption_is_rejected_for_base_v3(endpoint):
    client, requests, runtime = endpoint
    runtime.model_cfg.use_caption_condition = False
    response = client.post("/api/v1/synthesize", data={"text": "こんにちは。", "caption": "明るい声"})
    assert response.status_code == 400
    assert requests == []
    assert client.post("/api/v1/synthesize", data={"text": "こんにちは。"}).status_code == 200


def test_split_shares_reference_caption_and_generated_seed(endpoint):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", data={
        "text": "こんにちは。\r\nお元気ですか。", "ref_wav_path": "first.wav",
        "caption": "明るい声", "auto_split": "true", "chunk_silence_ms": "150",
    })
    assert response.status_code == 200
    assert [request.text for request in requests] == ["こんにちは。", "お元気ですか。"]
    assert [request.seed for request in requests] == [None, 677]
    assert requests[0].ref_wavs == requests[1].ref_wavs
    assert all(request.caption == "明るい声" for request in requests)
    assert sf.info(response.json()["candidates"][0]["path"]).frames == 480 * 2 + 7200


def test_oom_preserves_error_contract_and_can_retry(endpoint):
    client, _, runtime = endpoint
    original = runtime.synthesize

    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("test OOM")

    runtime.synthesize = fail
    assert client.post("/api/v1/synthesize", data={"text": "こんにちは。"}).status_code == 507
    runtime.synthesize = original
    assert client.post("/api/v1/synthesize", data={"text": "こんにちは。"}).status_code == 200


def test_health_and_output_path_restrictions(endpoint):
    client, _, _ = endpoint
    assert client.get("/api/v1/health").json() == {
        "status": "ok", "service": "irodori-tts", "default_checkpoint": "test-checkpoint",
    }
    assert client.get("/files/README.md").status_code == 404
    assert client.get("/files/sample_missing.wav").status_code == 404
