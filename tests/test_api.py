from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from irodori_tts.inference_runtime import SamplingRequest, SamplingResult
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
            [("sample_rf", 0.01)], 0.02, 677 if request.seed is None else request.seed, [],
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
        app, default_checkpoint="test-checkpoint", default_model_device="cuda",
        default_model_precision="fp32", default_codec_device="cuda", default_codec_precision="fp32",
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
    assert [Path(path).name for path in requests[0].ref_wavs] == ["second.wav", "first.wav"]
    assert requests[0].caption == "落ち着いた声"
    assert requests[0].num_steps is None


@pytest.mark.parametrize("caption", ["", "明るい声"])
def test_no_reference(endpoint, caption):
    client, requests, _ = endpoint
    response = client.post("/api/v1/synthesize", data={"text": "こんにちは。", "caption": caption})
    assert response.status_code == 200
    assert requests[0].no_ref
    assert requests[0].ref_wavs is None
    assert requests[0].cfg_scale_speaker == 5.0


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


@pytest.mark.parametrize("caption", ["", "落ち着いた女性が穏やかに話す"])
@pytest.mark.parametrize("seed", [None, 0, 677])
@pytest.mark.parametrize("speaker_scale", [0.0, 2.5])
def test_split_without_reference_reuses_first_audio(endpoint, caption, seed, speaker_scale):
    client, requests, runtime = endpoint
    original = runtime.synthesize
    reference_paths = []

    def generate(request, **kwargs):
        if requests:
            assert not request.no_ref
            path = Path(request.ref_wavs[0])
            reference_paths.append(path)
            reference, sample_rate = sf.read(path)
            assert sample_rate == 48000
            assert reference.shape == (480,)
            assert reference == pytest.approx(0.1, abs=1e-4)
        result = original(request, **kwargs)
        result.audio.mul_(len(requests))
        return result

    runtime.synthesize = generate
    data = {
        "text": "こんにちは。今日は晴れています。散歩に出かけましょう。",
        "caption": caption, "auto_split": "true", "cfg_scale_speaker": str(speaker_scale),
    }
    if seed is not None:
        data["seed"] = str(seed)
    response = client.post("/api/v1/synthesize", data=data)
    assert response.status_code == 200, response.text
    assert len(requests) == 3
    assert requests[0].no_ref and requests[0].ref_wavs is None
    assert len(reference_paths) == 2 and reference_paths[0] == reference_paths[1]
    assert not reference_paths[0].parent.exists()
    assert all(request.caption == (caption or None) for request in requests)
    assert all(request.cfg_scale_speaker == speaker_scale for request in requests[1:])
    used_seed = 677 if seed is None else seed
    assert [request.seed for request in requests] == [seed, used_seed, used_seed]
    assert response.json()["used_seed"] == used_seed
    audio, _ = sf.read(response.json()["candidates"][0]["path"])
    assert len(audio) == 480 * 3 + 7200 * 2
    assert audio[:480] == pytest.approx(0.1, abs=1e-4)
    assert audio[7680:8160] == pytest.approx(0.2, abs=1e-4)
    assert audio[-480:] == pytest.approx(0.3, abs=1e-4)


@pytest.mark.parametrize("text, speaker_condition", [
    ("こんにちは。", True),
    ("こんにちは。今日は晴れています。", False),
])
def test_split_does_not_create_unneeded_reference(endpoint, monkeypatch, text, speaker_condition):
    client, requests, runtime = endpoint
    runtime.model_cfg.use_speaker_condition_resolved = speaker_condition
    original = facade.save_wav
    saved_paths = []

    def save(path, *args):
        saved_paths.append(Path(path))
        return original(path, *args)

    monkeypatch.setattr(facade, "save_wav", save)
    response = client.post("/api/v1/synthesize", data={"text": text, "auto_split": "true"})
    assert response.status_code == 200
    assert all(request.no_ref and request.ref_wavs is None for request in requests)
    assert saved_paths == [Path(response.json()["candidates"][0]["path"])]


@pytest.mark.parametrize("reference", [
    {"ref_wavs": ["external.wav"]},
    {"ref_embed": "external.speaker.safetensors"},
    {"ref_latents": ["external.pt"]},
])
def test_split_preserves_explicit_speaker_condition(endpoint, monkeypatch, reference):
    _, requests, runtime = endpoint
    monkeypatch.setattr(facade, "save_wav", lambda *args: pytest.fail("generated reference saved"))
    facade.run_split_synthesis(
        chunks=["こんにちは。", "お元気ですか。"],
        build_request=lambda text, seed: SamplingRequest(text=text, seed=seed, **reference),
        initial_seed=0, runtime=runtime, log_fn=lambda message: None, silence_ms=0,
    )
    assert len(requests) == 2
    assert all(not request.no_ref for request in requests)
    for name, value in reference.items():
        assert all(getattr(request, name) == value for request in requests)


def test_split_removes_generated_reference_after_inference_failure(endpoint):
    client, requests, runtime = endpoint
    original = runtime.synthesize
    reference_paths = []

    def generate(request, **kwargs):
        if requests:
            path = Path(request.ref_wavs[0])
            assert path.is_file()
            reference_paths.append(path)
            raise torch.cuda.OutOfMemoryError("test OOM after first chunk")
        return original(request, **kwargs)

    runtime.synthesize = generate
    response = client.post("/api/v1/synthesize", data={
        "text": "こんにちは。お元気ですか。", "auto_split": "true",
    })
    assert response.status_code == 507, response.text
    assert len(reference_paths) == 1 and not reference_paths[0].parent.exists()
    runtime.synthesize = original
    assert client.post("/api/v1/synthesize", data={"text": "こんにちは。"}).status_code == 200


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
