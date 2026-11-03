from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_DLL_HANDLES: list[object] = []


def configure_runtime() -> None:
    os.environ.setdefault("IRODORI_REFERENCE_AUDIO_DIR", str(PROJECT_ROOT / "reference_audio"))
    os.environ.setdefault("IRODORI_API_OUTPUT_DIR", str(PROJECT_ROOT / "api_outputs" / "irodori-tts"))
    os.environ.setdefault("IRODORI_CODEC_DEVICE", "cpu")
    os.environ.setdefault("IRODORI_CODEC_PRECISION", "fp32")

    local_bin = PROJECT_ROOT / "tools" / "ffmpeg" / "bin"
    if local_bin.is_dir():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt":
        ffmpeg = shutil.which("ffmpeg")
        ffmpeg_bin = Path(ffmpeg).resolve().parent if ffmpeg else None
        if ffmpeg_bin is None or not any(ffmpeg_bin.glob("avcodec-*.dll")):
            raise SystemExit(
                "FFmpeg shared DLLs are required. Extract an FFmpeg 7 shared build into "
                "tools/ffmpeg/ so tools/ffmpeg/bin contains ffmpeg.exe and its DLLs."
            )
        # Python 3.8以降のDLL探索ではPATHの追加だけでは不十分なため。
        _DLL_HANDLES.append(os.add_dll_directory(str(ffmpeg_bin)))


def main() -> None:
    configure_runtime()
    from gradio_app import main as launch_ui

    launch_ui()


if __name__ == "__main__":
    main()
