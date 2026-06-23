FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# libsndfile1: soundfile, ffmpeg: torchcodec, git: dacvae install, pkg-config: sentencepiece build
# python3 + pip + libpython3.10: ベースに含まれないため追加。libpython3.10 は torchcodec の FFmpeg4 用 .so がリンクする libpython3.10.so.1.0 を提供
# PyTorch スタックは後段で cu130 wheel から導入する (Blackwell GB10 cc 12.1 SASS 対応のため)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git pkg-config \
        python3 python3-pip python3-venv libpython3.10 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY Irodori-TTS/ .
COPY gradio_app.py .
COPY gradio_app_voicedesign.py .

# pyproject.toml は cu128 インデックス固定のため、uv sync では PyTorch 一式をスキップし、
# cu130 wheel から別途投入する。cu130 ビルドは sm_121a SASS を含み GB10 (cc 12.1) を直接サポート。
# torchcodec も cu130 wheel に aarch64 版があるためここで一括投入する。
RUN uv sync --frozen --no-dev --extra cu128 \
        --no-install-package torch \
        --no-install-package torchaudio \
        --no-install-package torchcodec && \
    uv pip install --index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.10.0+cu130' 'torchaudio==2.10.0+cu130' 'torchcodec==0.10.0+cu130'

EXPOSE 7860
EXPOSE 7861

# GB10 (cc 12.1) は PyTorch 公式 wheel の get_arch_list() に未登録のため毎回出る互換性警告を抑制。
# cu130 wheel には sm_121a cubin が同梱されており、実行自体は問題なく動作する。
ENV PYTHONWARNINGS="ignore::UserWarning:torch.cuda"

ENTRYPOINT ["uv", "run", "--no-sync", "python"]
CMD ["gradio_app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
