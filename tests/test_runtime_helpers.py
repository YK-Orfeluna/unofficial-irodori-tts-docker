from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from irodori_tts.inference_runtime import RuntimeKey
from irodori_tts_api import runtime_helpers as helpers


def test_download_includes_tokenizer_and_quantized_subfolder(monkeypatch):
    sources = []
    monkeypatch.setattr(helpers, "download_hf_checkpoint", lambda source: sources.append(source) or "/cache/model.safetensors")
    source = "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only"
    assert helpers.resolve_checkpoint_path(source, log_prefix="test") == "/cache/model.safetensors"
    assert sources == [source]


@pytest.mark.parametrize("variant, capability", [
    ("int4-weight-only", "8.0"), ("float8-weight-only", "8.9"), ("float8-dynamic", "8.9"),
])
def test_turing_rejects_unsupported_quantization_before_download(monkeypatch, variant, capability):
    monkeypatch.setattr(helpers, "runtime_precision_choices", lambda device: ["fp32"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    def unexpected_download(*args, **kwargs):
        pytest.fail("unsupported checkpoint was downloaded")

    monkeypatch.setattr(helpers, "download_hf_checkpoint", unexpected_download)
    with pytest.raises(ValueError, match=capability):
        helpers.build_runtime_key(
            checkpoint=f"Aratako/Irodori-TTS-v4.1-Small-Quantized/{variant}",
            model_device="cuda", model_precision="fp32", codec_device="cpu",
            codec_precision="fp32", log_prefix="test",
        )


def test_local_quantization_metadata_is_validated(tmp_path, monkeypatch):
    path = tmp_path / "local.safetensors"
    save_file({"weight": torch.zeros(1)}, str(path), metadata={
        "irodori_quantization_json": json.dumps({
            "format_version": 1, "backend": "torchao", "quantization_type": "int4_weight_only",
        }),
    })
    with pytest.raises(ValueError, match="8.0"):
        helpers.build_runtime_key(
            checkpoint=str(path), model_device="cpu", model_precision="fp32",
            codec_device="cpu", codec_precision="fp32", log_prefix="test",
        )


def test_turing_precision_choices(monkeypatch):
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32", "bf16"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    assert helpers.runtime_precision_choices("cuda") == ["fp32"]


def test_model_switch_unloads_before_loading_and_reuses_current(monkeypatch):
    events = []
    monkeypatch.setattr(helpers, "_runtime_key", None)
    monkeypatch.setattr(helpers, "clear_cached_runtime", lambda: events.append("unload"))
    monkeypatch.setattr(helpers, "get_cached_runtime", lambda key: (events.append(key.checkpoint), True))
    first = RuntimeKey(checkpoint="first", model_device="cpu")
    second = RuntimeKey(checkpoint="second", model_device="cpu")
    helpers.load_runtime(first)
    helpers.load_runtime(first)
    helpers.load_runtime(second)
    assert events == ["unload", "first", "first", "unload", "second"]


def test_failed_model_load_can_recover(monkeypatch):
    monkeypatch.setattr(helpers, "_runtime_key", None)
    monkeypatch.setattr(helpers, "clear_cached_runtime", lambda: None)
    key = RuntimeKey(checkpoint="default", model_device="cpu")

    def fail(key):
        raise ValueError("invalid checkpoint")

    monkeypatch.setattr(helpers, "get_cached_runtime", fail)
    with pytest.raises(ValueError):
        helpers.load_runtime(key)
    assert helpers._runtime_key is None
    monkeypatch.setattr(helpers, "get_cached_runtime", lambda key: ("runtime", True))
    assert helpers.load_runtime(key) == ("runtime", True)


def test_ui_default_model_and_multiple_reference_inputs():
    import gradio_app

    assert gradio_app._default_checkpoint() == helpers.DEFAULT_CHECKPOINT
    assert gradio_app._resolve_ref_wavs("single.wav") == ["single.wav"]
    assert gradio_app._resolve_ref_wavs(["second.wav", {"path": "first.wav"}]) == ["second.wav", "first.wav"]
    assert gradio_app._resolve_ref_wavs(None) == []
    assert any("-Quantized/int8-weight-only" in value for value in gradio_app._checkpoint_choices())
    demo = gradio_app.build_ui()
    assert demo is not None
