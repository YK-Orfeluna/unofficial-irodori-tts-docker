FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# libsndfile1: soundfile, ffmpeg: torchcodec, git: dacvae install, pkg-config: sentencepiece build
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg git pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages uv

WORKDIR /app
COPY Irodori-TTS/ .

RUN uv sync --frozen --no-dev

EXPOSE 7860

ENTRYPOINT ["uv", "run", "python", "gradio_app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
