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
            model_device="cuda", model_precision="fp32", codec_device="cuda",
            codec_precision="fp32", log_prefix="test",
        )


def test_local_quantization_metadata_is_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    path = tmp_path / "local.safetensors"
    save_file({"weight": torch.zeros(1)}, str(path), metadata={
        "irodori_quantization_json": json.dumps({
            "format_version": 1, "backend": "torchao", "quantization_type": "int4_weight_only",
        }),
    })
    with pytest.raises(ValueError, match="8.0"):
        helpers.build_runtime_key(
            checkpoint=str(path), model_device="cuda", model_precision="fp32",
            codec_device="cuda", codec_precision="fp32", log_prefix="test",
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


@pytest.mark.parametrize("model_override, codec_override, model_expected, codec_expected", [
    (None, None, "bf16", "bf16"),
    ("fp32", None, "fp32", "bf16"),
    (None, "fp32", "bf16", "fp32"),
])
def test_ui_cuda_devices_and_precision(monkeypatch, model_override, codec_override, model_expected, codec_expected):
    import gradio_app

    monkeypatch.setenv("IRODORI_CODEC_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32", "bf16"])
    for name, override in [("IRODORI_MODEL_PRECISION", model_override), ("IRODORI_CODEC_PRECISION", codec_override)]:
        if override is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, override)
    demo = gradio_app.build_ui()
    blocks = {getattr(block, "label", None): block for block in demo.blocks.values()}
    assert blocks["Model Precision"].value == model_expected
    assert blocks["Codec Precision"].value == codec_expected
    for label in ["Model Device", "Codec Device"]:
        assert blocks[label].value == "cuda"
        assert blocks[label].choices == [("cuda", "cuda")]
        assert blocks[label].interactive is False


@pytest.mark.parametrize("model_device, codec_device", [
    ("cpu", "cuda"), ("cuda", "cpu"), ("cpu", "cpu"), ("mps", "cuda"), ("cuda", "xpu"),
])
def test_non_cuda_device_is_rejected_before_checkpoint_download(monkeypatch, model_device, codec_device):
    monkeypatch.setattr(helpers, "resolve_checkpoint_path", lambda *args, **kwargs: pytest.fail("checkpoint accessed"))
    with pytest.raises(ValueError, match="Both model_device and codec_device must use CUDA"):
        helpers.build_runtime_key(
            checkpoint=helpers.DEFAULT_CHECKPOINT, model_device=model_device,
            codec_device=codec_device, model_precision="fp32", codec_precision="fp32",
            log_prefix="test",
        )


def test_ui_does_not_fall_back_to_cpu_without_cuda(monkeypatch):
    import gradio_app

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("IRODORI_CODEC_DEVICE", "cpu")
    assert gradio_app._default_model_device() == "cuda"
    assert gradio_app._default_codec_device() == "cuda"
    with pytest.raises(ValueError, match="CUDA device requested"):
        gradio_app.build_ui()


def test_cuda_device_indices_are_kept(monkeypatch):
    monkeypatch.setattr(helpers, "runtime_precision_choices", lambda device: ["fp32"])
    monkeypatch.setattr(helpers, "resolve_checkpoint_path", lambda *args, **kwargs: "checkpoint_test.pt")
    key = helpers.build_runtime_key(
        checkpoint="checkpoint_test.pt", model_device="cuda:1", codec_device="cuda:0",
        model_precision="fp32", codec_precision="fp32", log_prefix="test",
    )
    assert key.model_device == "cuda:1"
    assert key.codec_device == "cuda:0"


def test_api_key_requires_available_cuda_before_download(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(helpers, "resolve_checkpoint_path", lambda *args, **kwargs: pytest.fail("checkpoint accessed"))
    with pytest.raises(ValueError, match="CUDA device requested"):
        helpers.build_runtime_key(
            checkpoint=helpers.DEFAULT_CHECKPOINT, model_device="cuda", codec_device="cuda",
            model_precision="fp32", codec_precision="fp32", log_prefix="test",
        )


def test_out_of_memory_does_not_retry_on_cpu(monkeypatch):
    key = RuntimeKey(checkpoint="test", model_device="cuda", codec_device="cuda")
    attempted = []
    monkeypatch.setattr(helpers, "_runtime_key", None)
    monkeypatch.setattr(helpers, "clear_cached_runtime", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fail(candidate):
        attempted.append(candidate)
        raise torch.cuda.OutOfMemoryError("test OOM")

    monkeypatch.setattr(helpers, "get_cached_runtime", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        helpers.load_runtime(key)
    assert attempted == [key]
    assert helpers._runtime_key is None


def test_compose_uses_shared_cuda_defaults():
    from pathlib import Path
    import yaml

    compose = yaml.safe_load((Path(__file__).resolve().parent.parent / "docker-compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["irodori-tts"]
    assert service["command"][0] == "gradio_app.py"
    environment = service.get("environment", {})
    assert environment.get("IRODORI_CODEC_DEVICE", "cuda") == "cuda"
    assert "IRODORI_MODEL_PRECISION" not in environment
    assert "IRODORI_CODEC_PRECISION" not in environment


@pytest.mark.parametrize("capability, expected", [((7, 5), "fp32"), ((8, 6), "bf16")])
def test_api_startup_uses_cuda_for_both_devices(monkeypatch, capability, expected):
    from types import SimpleNamespace
    import gradio_app
    from irodori_tts_api import app as api

    attached = {}
    monkeypatch.setattr("sys.argv", ["gradio_app.py", "--server-name", "0.0.0.0"])
    monkeypatch.setenv("IRODORI_CODEC_DEVICE", "cpu")
    monkeypatch.delenv("IRODORI_MODEL_PRECISION", raising=False)
    monkeypatch.delenv("IRODORI_CODEC_PRECISION", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(helpers, "list_available_runtime_precisions", lambda device: ["fp32", "bf16"])
    monkeypatch.setattr(gradio_app, "build_ui", lambda: SimpleNamespace(queue=lambda **kwargs: None))
    monkeypatch.setattr(gradio_app.gr, "mount_gradio_app", lambda app, *args, **kwargs: app)
    monkeypatch.setattr(api, "attach_api", lambda app, **kwargs: attached.update(kwargs))
    monkeypatch.setattr(gradio_app.uvicorn, "Server", lambda config: SimpleNamespace(started=True, run=lambda: None))
    gradio_app.main()
    assert attached["default_model_device"] == "cuda"
    assert attached["default_codec_device"] == "cuda"
    assert attached["default_model_precision"] == expected
    assert attached["default_codec_precision"] == expected


def test_local_checkpoint_search_is_relative_to_project(tmp_path, monkeypatch):
    import gradio_app

    project = tmp_path / "日本語 project"
    project.mkdir()
    checkpoint = project / "checkpoint_test.pt"
    checkpoint.touch()
    monkeypatch.setattr(gradio_app, "__file__", str(project / "gradio_app.py"))
    monkeypatch.chdir(tmp_path)
    assert gradio_app._local_checkpoint_candidates() == [str(checkpoint)]
