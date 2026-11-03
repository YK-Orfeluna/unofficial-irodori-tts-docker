from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows command tests")
SOURCE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "日本語 project"
    root.mkdir()
    for name in ("setup.cmd", "start.cmd"):
        shutil.copyfile(SOURCE_ROOT / name, root / name)
    return root


def run_cmd(root, name, *args):
    env = dict(os.environ)
    env["PATH"] = str(Path(env["SYSTEMROOT"]) / "System32")
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(root / name), *args],
        cwd=root.parent, env=env, stdin=subprocess.DEVNULL, capture_output=True,
        encoding="utf-8", errors="replace", timeout=120,
    )


def test_start_without_setup_shows_instruction(project):
    result = run_cmd(project, "start.cmd")
    assert result.returncode == 1
    assert "Run setup.cmd first" in result.stdout


def test_setup_propagates_failure_and_handles_spaces(project):
    scripts = project / "scripts"
    scripts.mkdir()
    (scripts / "bootstrap_uv.ps1").write_text("Write-Host 'fixture-setup-failed'\nexit 7\n", encoding="ascii")
    result = run_cmd(project, "setup.cmd")
    assert result.returncode == 7
    assert "fixture-setup-failed" in result.stdout
    assert "Setup failed" in result.stdout


def test_start_passes_arguments_and_runs_in_project(project):
    venv = project / ".venv"
    (venv / "Scripts").mkdir(parents=True)
    shutil.copyfile(sys.executable, venv / "Scripts/python.exe")
    shutil.copyfile(Path(sys.prefix) / "pyvenv.cfg", venv / "pyvenv.cfg")
    (project / "main.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "Path('invoked.json').write_text(json.dumps({'cwd':str(Path.cwd()),'args':sys.argv[1:]}))\n",
        encoding="ascii",
    )
    result = run_cmd(project, "start.cmd", "--server-port", "17862")
    assert result.returncode == 0, result.stdout + result.stderr
    invoked = json.loads((project / "invoked.json").read_text())
    assert invoked == {"cwd": str(project), "args": ["--open-browser", "--server-port", "17862"]}


def test_bootstrap_download_and_extraction_without_external_install(project):
    scripts = project / "scripts"
    scripts.mkdir()
    shutil.copyfile(SOURCE_ROOT / "scripts/bootstrap_uv.ps1", scripts / "bootstrap_uv.ps1")
    (scripts / "windows_setup.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('invoked.txt').write_text(sys.argv[-1])\n", encoding="ascii",
    )
    uv = SOURCE_ROOT / "tools/uv/uv.exe"
    if not uv.exists():
        uv = Path(shutil.which("uv") or Path.home() / ".local/bin/uv.exe")
    assert uv.is_file(), "uv is required to run the Windows bootstrap test"
    archive = project / "uv-fixture.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.write(uv, "uv.exe")
    wrapper = project / "invoke.ps1"
    wrapper.write_text(
        "function Invoke-WebRequest { param($Uri, $OutFile, [switch]$UseBasicParsing)\n"
        "Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'uv-fixture.zip') -Destination $OutFile\n}\n"
        "& (Join-Path $PSScriptRoot 'scripts/bootstrap_uv.ps1')\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (scripts / "invoked.txt").read_text(encoding="utf-8") == str(project / "tools/uv/uv.exe")
