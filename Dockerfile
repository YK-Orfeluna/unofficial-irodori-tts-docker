FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# torchcodec の FFmpeg4 バックエンドが libpython3.10.so.1.0 を必要とする。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git pkg-config \
        python3 python3-pip python3-venv libpython3.10 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY Irodori-TTS/ .

# UI/APIの修正で依存関係を再取得しないよう、別レイヤーにする。
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra cu128

COPY gradio_app.py .
COPY irodori_tts_api/ ./irodori_tts_api/

EXPOSE 7860

ENTRYPOINT ["uv", "run", "--no-sync", "python"]
CMD ["gradio_app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
