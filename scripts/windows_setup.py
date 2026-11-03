from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_REVISION = "89f9d8fbd4d51ea019867ee1197725ede1df13c5"
GIT_URL = "https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.4/MinGit-2.56.0.4-64-bit.zip"
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-win64-lgpl-shared-7.1.zip"
UPSTREAM_URL = f"https://github.com/Aratako/Irodori-TTS/archive/{UPSTREAM_REVISION}.zip"


def install_archive(url: str, target: Path, *, nested: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        work = Path(temporary)
        archive = work / "download.zip"
        urllib.request.urlretrieve(url, archive)
        extracted = work / "extracted"
        with zipfile.ZipFile(archive) as package:
            package.extractall(extracted)
        content = next(extracted.iterdir()) if nested else extracted
        shutil.copytree(content, target, dirs_exist_ok=True)


def setup(uv: Path) -> None:
    tools = PROJECT_ROOT / "tools"
    git = tools / "git" / "cmd" / "git.exe"
    if not git.is_file():
        print("Gitを準備しています...", flush=True)
        install_archive(GIT_URL, tools / "git")
    env = dict(os.environ)
    env["PATH"] = str(git.parent) + os.pathsep + env.get("PATH", "")

    upstream = PROJECT_ROOT / "Irodori-TTS"
    if (PROJECT_ROOT / ".git").exists():
        print("上流コードを確認しています...", flush=True)
        subprocess.run(
            [str(git), "submodule", "update", "--init", "--recursive"],
            cwd=PROJECT_ROOT, env=env, check=True,
        )
    elif not (upstream / "pyproject.toml").is_file():
        print("上流コードを取得しています...", flush=True)
        install_archive(UPSTREAM_URL, upstream, nested=True)

    ffmpeg = tools / "ffmpeg"
    if not all((ffmpeg / "bin" / name).is_file() for name in ("ffmpeg.exe", "ffprobe.exe", "avcodec-61.dll")):
        print("FFmpegを準備しています...", flush=True)
        install_archive(FFMPEG_URL, ffmpeg, nested=True)

    print("Pythonパッケージを準備しています。初回は時間がかかります...", flush=True)
    subprocess.run([str(uv), "sync", "--locked", "--no-dev"], cwd=PROJECT_ROOT, env=env, check=True)
    subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"), "-c",
         "from main import configure_runtime; configure_runtime(); import torch, torchcodec, gradio_app; "
         "assert torch.cuda.is_available(), 'NVIDIA GPU or compatible driver was not found'; "
         "print('GPU:', torch.cuda.get_device_name(0))"],
        cwd=PROJECT_ROOT, env=env, check=True,
    )
    print("セットアップが完了しました。start.cmd を実行してください。", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", required=True, type=Path)
    args = parser.parse_args()
    try:
        setup(args.uv)
    except (OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"セットアップに失敗しました: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
