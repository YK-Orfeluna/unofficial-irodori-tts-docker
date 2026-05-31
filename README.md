# unofficial-irodori-tts-docker

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v3 の推論 UI を Docker で手軽に動かすための非公式ラッパーです。

> **免責事項**
> このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の非公式ラッパーです。元リポジトリの作者とは無関係であり、公式サポートを提供するものではありません。

## 概要

- Irodori-TTS v3 を Git submodule として管理
- `docker compose up --build` の1コマンドで Gradio UI を起動
- モデルは初回起動時に HuggingFace Hub から自動ダウンロード

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

起動後、以下の URL から各 UI にアクセスできます。

| UI | URL | 対応モデル |
|---|---|---|
| 500M モデル | `http://localhost:7860` | `Irodori-TTS-500M-v2 / v3` |
| VoiceDesign モデル | `http://localhost:7861` | `Irodori-TTS-600M-VoiceDesign-v2 / v3` |

## モデルについて

初回起動時に選択したモデルが `~/.cache/huggingface/` にダウンロードされます。2回目以降はキャッシュを利用するため再ダウンロードは不要です。

## クレジット

- 元リポジトリ: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- ライセンス: 元リポジトリのライセンスに従います
