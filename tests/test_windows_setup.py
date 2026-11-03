from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts import windows_setup as setup


@pytest.mark.parametrize("nested", [False, True])
def test_archive_installs_and_preserves_existing_files(tmp_path, monkeypatch, nested):
    archive = tmp_path / "fixture.zip"
    prefix = "release/" if nested else ""
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(prefix + "bin/tool.exe", b"tool")
    monkeypatch.setattr(setup.urllib.request, "urlretrieve", lambda url, path: shutil.copyfile(archive, path))
    target = tmp_path / "tools" / "installed"
    target.mkdir(parents=True)
    (target / "existing").write_text("keep")
    setup.install_archive("https://example.test/tool.zip", target, nested=nested)
    assert (target / "bin/tool.exe").read_bytes() == b"tool"
    assert (target / "existing").read_text() == "keep"


def test_broken_archive_does_not_create_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"broken"))
    target = tmp_path / "tool"
    with pytest.raises(zipfile.BadZipFile):
        setup.install_archive("https://example.test/tool.zip", target)
    assert not target.exists()


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "日本語 project"
    root.mkdir()
    monkeypatch.setattr(setup, "PROJECT_ROOT", root)
    installed = []
    commands = []

    def install(url, target, **kwargs):
        installed.append(url)
        files = {
            setup.GIT_URL: ["cmd/git.exe"],
            setup.FFMPEG_URL: ["bin/ffmpeg.exe", "bin/ffprobe.exe", "bin/avcodec-61.dll"],
            setup.UPSTREAM_URL: ["pyproject.toml"],
        }[url]
        for name in files:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def run(args, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] == root
        assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(root / "tools/git/cmd")
        commands.append(args)

    monkeypatch.setattr(setup, "install_archive", install)
    monkeypatch.setattr(setup.subprocess, "run", run)
    return root, installed, commands


def test_zip_download_needs_no_preinstalled_git_or_python(project):
    root, installed, commands = project
    previous_path = os.environ.get("PATH")
    setup.setup(root / "tools/uv/uv.exe")
    assert installed == [setup.GIT_URL, setup.UPSTREAM_URL, setup.FFMPEG_URL]
    assert commands[0][1:] == ["sync", "--locked", "--no-dev"]
    assert commands[1][0] == str(root / ".venv/Scripts/python.exe")
    assert os.environ.get("PATH") == previous_path


def test_git_checkout_initializes_submodule_and_can_be_repeated(project):
    root, installed, commands = project
    (root / ".git").mkdir()
    setup.setup(root / "uv.exe")
    setup.setup(root / "uv.exe")
    assert installed == [setup.GIT_URL, setup.FFMPEG_URL]
    assert commands[0][1:] == ["submodule", "update", "--init", "--recursive"]
    assert commands[3][1:] == ["submodule", "update", "--init", "--recursive"]


def test_zip_with_upstream_already_present_is_reused(project):
    root, installed, _ = project
    (root / "Irodori-TTS").mkdir()
    (root / "Irodori-TTS/pyproject.toml").touch()
    setup.setup(root / "uv.exe")
    assert setup.UPSTREAM_URL not in installed


def test_dependency_failure_stops_before_success(project, monkeypatch, capsys):
    root, _, _ = project

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(setup.subprocess, "run", fail)
    monkeypatch.setattr("sys.argv", ["windows_setup.py", "--uv", str(root / "uv.exe")])
    assert setup.main() == 1
    output = capsys.readouterr().out
    assert "セットアップに失敗" in output
    assert "セットアップが完了" not in output
