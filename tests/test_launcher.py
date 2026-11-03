from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import main as launcher


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.os, "environ", os.environ.copy())
    root = tmp_path / "日本語 project"
    root.mkdir()
    (root / "avcodec-61.dll").touch()
    monkeypatch.setattr(launcher, "PROJECT_ROOT", root)
    for name in (
        "IRODORI_REFERENCE_AUDIO_DIR", "IRODORI_API_OUTPUT_DIR",
        "IRODORI_CODEC_DEVICE", "IRODORI_CODEC_PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    return root


def test_launcher_paths_do_not_depend_on_working_directory(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: str(project / "ffmpeg.exe"))
    if os.name == "nt":
        monkeypatch.setattr(launcher.os, "add_dll_directory", lambda path: object())
    launcher.configure_runtime()
    assert os.environ["IRODORI_REFERENCE_AUDIO_DIR"] == str(project / "reference_audio")
    assert os.environ["IRODORI_API_OUTPUT_DIR"] == str(project / "api_outputs" / "irodori-tts")
    assert "IRODORI_CODEC_DEVICE" not in os.environ
    assert "IRODORI_CODEC_PRECISION" not in os.environ


def test_launcher_keeps_explicit_configuration(project, monkeypatch):
    monkeypatch.setenv("IRODORI_API_OUTPUT_DIR", str(project / "custom"))
    monkeypatch.setenv("IRODORI_CODEC_PRECISION", "fp32")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: str(project / "ffmpeg.exe"))
    if os.name == "nt":
        monkeypatch.setattr(launcher.os, "add_dll_directory", lambda path: object())
    launcher.configure_runtime()
    assert os.environ["IRODORI_API_OUTPUT_DIR"] == str(project / "custom")
    assert os.environ["IRODORI_CODEC_PRECISION"] == "fp32"


def test_windows_registers_local_ffmpeg_dll_directory(project, monkeypatch):
    bin_dir = project / "tools" / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "avcodec-61.dll").touch()
    registered = []
    handle = object()
    monkeypatch.setattr(launcher, "_DLL_HANDLES", [])
    monkeypatch.setenv("PATH", "original-path")
    monkeypatch.setattr(launcher, "os", SimpleNamespace(
        environ=os.environ, name="nt", pathsep=os.pathsep,
        add_dll_directory=lambda path: registered.append(path) or handle,
    ))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: str(bin_dir / "ffmpeg.exe"))
    launcher.configure_runtime()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert registered == [str(bin_dir)]
    assert launcher._DLL_HANDLES == [handle]


def test_windows_missing_ffmpeg_reports_install_location(project, monkeypatch):
    monkeypatch.setattr(launcher, "os", SimpleNamespace(environ=os.environ, name="nt"))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="tools/ffmpeg/bin"):
        launcher.configure_runtime()


def test_windows_static_ffmpeg_reports_shared_build_requirement(project, monkeypatch):
    monkeypatch.setattr(launcher, "os", SimpleNamespace(environ=os.environ, name="nt"))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: str(project / "static" / "ffmpeg.exe"))
    with pytest.raises(SystemExit, match="FFmpeg shared DLLs are required"):
        launcher.configure_runtime()
