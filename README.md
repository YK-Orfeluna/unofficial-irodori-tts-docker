# unofficial-irodori-tts-docker

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v2 の推論 UI を Docker で手軽に動かすための非公式ラッパーです。

> **免責事項**
> このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の非公式ラッパーです。元リポジトリの作者とは無関係であり、公式サポートを提供するものではありません。

## 概要

- Irodori-TTS v2 を Git submodule として管理
- `docker compose up --build` の1コマンドで Gradio UI を起動
- モデルは初回起動時に HuggingFace Hub から自動ダウンロード（ローカル配置も可）

## 前提条件

- Docker / Docker Compose
- NVIDIA GPU
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## セットアップ

```bash
git clone --recurse-submodules https://github.com/YK-Orfeluna/unofficial-irodori-tts-docker.git
cd unofficial-irodori-tts-docker
docker compose up --build
```

ブラウザで `http://localhost:7860` を開くと UI が起動しています。

## モデルについて

**HuggingFace Hub から自動ダウンロード（デフォルト）**

初回起動時に `Aratako/Irodori-TTS-500M-v2` が `~/.cache/huggingface/` にダウンロードされます。2回目以降はキャッシュを利用するため再ダウンロードは不要です。

**ローカルモデルを使う場合**

`./pretrained_models/` にチェックポイントファイルを配置すると自動検出されます。

```
pretrained_models/
└── checkpoint_xxxxx.safetensors
```

## クレジット

- 元リポジトリ: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- ライセンス: 元リポジトリのライセンスに従います
