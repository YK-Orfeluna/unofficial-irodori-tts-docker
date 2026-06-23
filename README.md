# unofficial-irodori-tts-docker

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v3 を Docker で動かすための非公式ラッパー。NVIDIA GB10 (Blackwell, cc 12.1) を含む aarch64 ホストで動作するよう、CUDA 13.0 + cu130 PyTorch wheel ベースに構成。

> **免責事項**
> このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の非公式ラッパーです。元リポジトリの作者とは無関係で、公式サポートは提供しません。

## 構成

- Irodori-TTS v3 を Git submodule (`Irodori-TTS/`) として管理
- Gradio UI 2 種 (Reference Audio / VoiceDesign) を Docker Compose profiles で切替
- 同じプロセスに FastAPI ベースの REST API (`irodori_tts_api/`) を mount。Gradio UI と同一 port で `/api/v1/*` を公開
- 外部ネットワーク `dify_default` (compose 上の alias は `dify_net`) に `irodori-tts-500m` / `irodori-tts-voicedesign` の名前で参加。Dify など同ネット上のサービスから上記ホスト名で参照可能
- 全 service に `/api/v1/health` の healthcheck 付き(初回起動を待つため `start_period: 15m`)

## 前提条件

- Docker / Docker Compose v2
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 推奨: NVIDIA GB10 (DGX Spark) など sm_121a (cu130) で動く GPU。他 GPU でも cu130 wheel が対応していれば動作可
- 外部ネットワーク `dify_default` が事前作成済(Dify の compose 起動済み等)

## セットアップ

```bash
git clone --recurse-submodules https://github.com/YK-Orfeluna/unofficial-irodori-tts-docker.git
cd unofficial-irodori-tts-docker

# 500M Reference Audio (port 7860)
docker compose --profile ref up --build -d

# VoiceDesign (port 7861)
docker compose --profile voicedesign up --build -d
```

両方同時に起動するなら `--profile ref --profile voicedesign` を併用。

| profile | UI / API URL | 内部 service 名 | 想定 checkpoint |
|---|---|---|---|
| `ref` | `http://localhost:7860` | `irodori-tts-500m` | `Aratako/Irodori-TTS-500M-v3` |
| `voicedesign` | `http://localhost:7861` | `irodori-tts-voicedesign` | VoiceDesign 系 checkpoint |

UI 用エンドポイント: `/` (Gradio)、API ドキュメント: `/api/docs`。

## モデルキャッシュ

初回起動時に Gradio UI / API 経由で選択した checkpoint が `~/.cache/huggingface/` にダウンロードされ、コンテナにマウントされます。2 回目以降はキャッシュを利用。

## 参照音声 (`reference_audio/`)

ディレクトリは `.gitkeep` のみ追跡し、音声ファイル本体は git 追跡外(`.gitignore`)。

実音声の配置はデプロイ側で行う想定です(例: 親リポジトリ `AI_Shibutani/scripts/sync_reference_audio.sh` から `AI_Shibutani/assets/reference_audio/` をコピー)。コンテナ内では `/app/reference_audio` に read-only マウントされ、REST API の `ref_wav_path` から参照できます。

API は traversal 防止のため `ref_wav_path` を `/app/reference_audio` 配下に限定します。

## 長文 auto_split

Diffusion ベース TTS は長文で後半が崩れやすいため、句読点優先の chunk 分割 → 順次合成 → 無音連結を opt-in で選べます。

### Gradio UI

"Sampling" アコーディオン内の以下のコントロール:

| コントロール | 既定 | 範囲 | 役割 |
|---|---|---|---|
| Auto Split | OFF | bool | opt-in スイッチ |
| Max Chars / Chunk | 200 | 50–500 | 1 chunk の最大文字数 |
| Chunk Silence (ms) | 150 | 0–1000 | chunk 間の無音長 |

### REST API パラメタ(multipart/form-data)

`auto_split=true`、`max_chars_per_chunk` (既定 200)、`chunk_silence_ms` (既定 150) を追加可能。

### 制約と挙動

- `auto_split=true` 時は `num_candidates=1` を強制(chunk × candidate の組み合わせが意味不明になるため)
- 全 chunk で同一 seed / ref_wav / CFG を共有して声質の一貫性を担保
- 1 chunk 目の `used_seed` 確定後、それを後続 chunk に流用

### 分割アルゴリズム(`irodori_tts_api/text_splitter.py`)

1. 改行を正規化(CRLF→LF、CR→LF、連続改行を1つに、先頭末尾の改行と周囲空白を除去)。「文字と文字に挟まれた改行」だけが分割キーとして残る
2. 句点 `。．！？!?` で文単位に
3. 超過する文は読点 `、,` で再分割
4. それでも超えるなら `max_chars` で強制カット
5. 極端に短い断片(3 文字未満)は直前 chunk に併合

## 改行の取り扱い

クライアント側(multipart parser 経由など)で `\r` / `\n` が紛れ込んでも合成音が乱れないよう、以下のレイヤで吸収:

| 経路 | 動作 |
|---|---|
| `auto_split=true` (UI / API 共通) | `text_splitter` が改行を正規化。中間改行は分割キーとして保持、ノイズ改行(先頭末尾・連続・単独 CR)は除去 |
| `auto_split=false` (UI / API 共通) | 合成呼び出し直前で `\r` / `\n` を空白に置換し、TTS に改行リテラルが渡らないようにする |

## REST API

`POST /api/v1/synthesize` を `ref` / `voicedesign` の各 profile で公開(同じパスだが別 service)。Reference 側のサンプル:

```bash
curl -X POST http://localhost:7860/api/v1/synthesize \
  -F text='こんにちは。私の名前はシブタニです。' \
  -F ref_wav_path=/app/reference_audio/220628_Shibutani-P.cut.wav \
  -F auto_split=true \
  -F seed=677
```

成功時 JSON 例:

```json
{
  "service": "irodori-tts-500m",
  "sample_rate": 48000,
  "used_seed": 677,
  "candidates": [
    {"index": 1, "path": "/app/api_outputs/sample_..._001.wav", "url": "/files/sample_..._001.wav"}
  ],
  "stage_timings": [["chunk_1/sample_rf", 0.42], ["chunk_2/sample_rf", 0.45]],
  "total_to_decode": 1.20,
  "messages": ["split into 2 chunks (silence=150ms)", "..."]
}
```

`url` フィールドは `GET /files/{name}` で取得可能(`/app/api_outputs/` 配下のみ、ファイル名は `sample_*.wav` パターンに限定)。

### エンドポイント一覧

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/api/v1/health` | service 名と default_checkpoint を返す |
| `POST` | `/api/v1/synthesize` | 合成(profile によって挙動が異なる) |
| `GET` | `/files/{name}` | 生成 wav の取得 |
| `GET` | `/api/docs` | Swagger UI |
| `GET` | `/api/openapi.json` | OpenAPI スキーマ |
| `GET` | `/` | Gradio UI |

## テスト

```bash
python3 -m pytest tests/
```

`tests/test_text_splitter.py` で改行正規化と分割ロジックを 14 件カバー(Dify 由来の `\r\n` 混入、単独 CR、先頭末尾改行のノイズ除去を含む)。

## クレジット

- 元リポジトリ: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- ライセンス: 元リポジトリのライセンスに従います
