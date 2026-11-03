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


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_supported_gpus_prefer_bf16(monkeypatch, capability):
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32", "bf16"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert helpers.runtime_precision_choices("cuda") == ["bf16", "fp32"]


def test_cpu_precision_choices(monkeypatch):
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32"])
    assert helpers.runtime_precision_choices("cpu") == ["fp32"]


def test_explicit_precision_overrides_default(monkeypatch):
    monkeypatch.setattr(helpers, "runtime_precision_choices", lambda device: ["bf16", "fp32"])
    monkeypatch.setenv("IRODORI_MODEL_PRECISION", "fp32")
    assert helpers.resolve_precision_from_env(
        "IRODORI_MODEL_PRECISION", device="cuda", fallback="bf16",
    ) == "fp32"


@pytest.mark.parametrize("value", ["bf16", "fp16", "invalid"])
def test_invalid_precision_is_rejected(monkeypatch, value):
    monkeypatch.setattr(helpers, "runtime_precision_choices", lambda device: ["fp32"])
    monkeypatch.setenv("IRODORI_MODEL_PRECISION", value)
    with pytest.raises(SystemExit) as error:
        helpers.resolve_precision_from_env(
            "IRODORI_MODEL_PRECISION", device="cuda", fallback="fp32",
        )
    assert error.value.code == 2


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_default_precision_is_logged(monkeypatch, capsys, precision):
    monkeypatch.delenv("IRODORI_MODEL_PRECISION", raising=False)
    assert helpers.resolve_precision_from_env(
        "IRODORI_MODEL_PRECISION", device="cuda", fallback=precision,
    ) == precision
    assert ("BF16 preferred" if precision == "bf16" else "BF16 unavailable") in capsys.readouterr().out


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


@pytest.mark.parametrize("override, expected", [(None, "bf16"), ("fp32", "fp32")])
def test_ui_initial_precision_and_device_change(monkeypatch, override, expected):
    import gradio_app

    monkeypatch.setattr(gradio_app, "_default_model_device", lambda: "cuda")
    monkeypatch.setattr(gradio_app, "_default_codec_device", lambda: "cpu")
    monkeypatch.setattr(gradio_app, "list_available_runtime_devices", lambda: ["cuda", "cpu"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32", "bf16"] if device == "cuda" else ["fp32"])
    monkeypatch.delenv("IRODORI_CODEC_PRECISION", raising=False)
    if override is None:
        monkeypatch.delenv("IRODORI_MODEL_PRECISION", raising=False)
    else:
        monkeypatch.setenv("IRODORI_MODEL_PRECISION", override)
    demo = gradio_app.build_ui()
    values = {getattr(block, "label", None): getattr(block, "value", None) for block in demo.blocks.values()}
    assert values["Model Precision"] == expected
    assert values["Codec Precision"] == "fp32"
    assert gradio_app._on_model_device_change("cuda").value == "bf16"


def test_local_checkpoint_search_is_relative_to_project(tmp_path, monkeypatch):
    import gradio_app

    project = tmp_path / "日本語 project"
    project.mkdir()
    checkpoint = project / "checkpoint_test.pt"
    checkpoint.touch()
    monkeypatch.setattr(gradio_app, "__file__", str(project / "gradio_app.py"))
    monkeypatch.chdir(tmp_path)
    assert gradio_app._local_checkpoint_candidates() == [str(checkpoint)]
