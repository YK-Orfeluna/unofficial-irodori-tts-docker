# unofficial-irodori-tts-docker

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4/v4.1 をDockerで動かす非公式ラッパーです。既定モデルは `Aratako/Irodori-TTS-v4.1-Small`。参照音声によるVoice CloningとcaptionによるVoiceDesignを、1つのGradio UI / REST APIに統合しています。

元リポジトリの作者とは無関係で、公式サポートは提供しません。

## セットアップ

Docker / Docker Compose v2、NVIDIA GPU、nvidia-container-toolkit、外部ネットワーク `dify_default` が必要です。CUDA 12.8と上流のcu128依存構成を使用します。

```bash
git clone --recurse-submodules https://github.com/YK-Orfeluna/unofficial-irodori-tts-docker.git
cd unofficial-irodori-tts-docker
docker compose up --build -d
```

- UI: `http://localhost:7860/`
- APIドキュメント: `http://localhost:7860/api/docs`
- health: `http://localhost:7860/api/v1/health`
- Difyネットワーク内: `irodori-tts-500m:7860` / `irodori-tts-voicedesign:7860`

Compose profilesは不要です。旧VoiceDesignの7861は廃止したため、クライアントの接続先を7860へ変更してください。旧DNS名2つは維持します。レスポンスの `service` 値は `irodori-tts` へ統一しています。

## モデル・精度・量子化

UIのCheckpointで通常モデル・量子化モデル・ローカルcheckpointを選べます。任意のモデルIDやローカルパスも入力できます。APIでは既存の `checkpoint` フィールドを使います。

既定のv4.1-Smallのほか、v4-Small、v4.1-Small-MF、v4-Large、v2/v3のbase・VoiceDesign、および公式Small/Large量子化モデルを候補に含めています。ローカルに旧checkpointが残っていても、起動時はv4.1-Smallを選択します。モデル変更時は旧モデルを先に解放します。

### v3からv4への変更点

[公式v3 README](https://github.com/Aratako/Irodori-TTS/blob/v3/README.md)と[公式v4/v4.1 README](https://github.com/Aratako/Irodori-TTS/blob/main/README.md)の記載を比較しています（2026-10-11確認）。v4の基本構成と、その後のv4.1・MeanFlow・量子化への拡張を含みます。

| 項目 | v3 READMEの記載 | v4/v4.1 READMEでの変更 |
|---|---|---|
| モデル構成 | 500Mのbaseと600MのVoiceDesignが別checkpoint。VoiceDesignはtext・参照音声・captionの3条件に対応 | v4-Small系の1つのcheckpointにbaseとVoiceDesignの機能を統合 |
| テキスト・captionのエンコーダ | テキストは事前学習LLM由来の埋め込みとTransformer層。captionには別エンコーダを使用 | fine-tuneしたModernBERTを両者で共有し、別々のプロジェクタで各条件へ変換 |
| 参照音声 | 単一の参照音声・latent・Speaker Inversion embeddingを使う例を記載。複数参照や長時間参照の説明はない | 複数clipを指定順に連結し、v4-Smallでは合計最大120秒。各clipを個別にエンコードしてからlatentを連結 |
| 少ステップ生成 | RFのEulerステップ削減とSway Samplingを紹介 | それらに加え、v4.1-Smallを蒸留したMeanFlow版を紹介。既定4ステップで、条件エンコーダ・音声長予測器は固定してDiTを学習 |
| 量子化 | 量子化checkpointの説明はない | torchaoによるINT8・INT4・FP8版と量子化スクリプトを紹介。対象層やGPU要件は下記の量子化節を参照 |
| checkpointの配布・移動 | ローカルcheckpointまたはHFモデルIDを指定する推論を紹介 | 事前学習テキストエンコーダを持つcheckpointの変換ではtokenizerディレクトリも生成し、重みと一緒に配置することを明記 |
| 旧モデルとの互換性 | v2のbase／VoiceDesignに対応 | v2・v3のbase／VoiceDesignを引き続き推論可能。旧コードはv3・v2・v1タグから参照可能 |

参照長の上限はcheckpointごとの仕様です。v3 READMEには上限秒数の記載がないため、この比較ではv3の上限値を推定していません。v4の長時間参照は、同じ話者の短いclipを複数組み合わせる構成が推奨され、評価では合計約30秒で改善の大部分が得られたとされています。単一の長い録音も入力できますが、その形式の効果は未評価です。[公式READMEの複数参照説明](https://github.com/Aratako/Irodori-TTS/blob/main/README.md#multiple-reference-clips)

**v3から継続する機能**は、自動音声長予測・音声長の倍率指定、VoiceDesignでのcaptionと参照音声の併用、絵文字によるスタイル制御、48kHzのDACVAEコーデック、SilentCipherによる透かし、LoRA、Speaker Inversionです。これらはv4で初めて追加された機能ではありません。v1のcheckpoint・前処理がv2/v3/v4と非互換である点も継続しています。

上流のモデル統合と、このラッパーのUI/API統合は別の変更です。上流READMEには現在もGradio UIを7860と7861で起動する例があり、両方の既定モデルがv4.1-Smallになっています。このラッパーはUI/APIを7860へ統合しています。

### v4系モデルの違い

[上流README](https://github.com/Aratako/Irodori-TTS)、[MeanFlowガイド](https://github.com/Aratako/Irodori-TTS/blob/main/docs/meanflow.md)、作者の公式モデルカードで確認できる内容をまとめています（2026-10-11確認）。

以下の通常モデルは、日本語のVoice Cloning・captionによるVoiceDesign・両者の併用・絵文字によるスタイル制御に対応します。参照音声は単一または複数、合計最大120秒。音声長の自動予測と48kHz出力も共通です。

| モデル（公式モデルカード） | 主な違い | ステップ既定 | 重み用VRAM概算：FP32 | 重み用VRAM概算：BF16 | モデルのライセンス表記 |
|---|---|---|---|---|---|
| [v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small) | 約766Mパラメータ。ModernBERT-jaをテキスト・caption共通エンコーダに採用したv4の基本モデル | RF 40 | 2.85 GiB | 1.43 GiB | MIT |
| [v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | 約766Mパラメータ。v4-Smallの音声長予測器だけを置換・再学習。長さの過大推定に起因する生成エラーを減らす更新。RF-DiTと各条件エンコーダは同じ | RF 40 | 2.85 GiB | 1.43 GiB | MIT |
| [v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | 約773Mパラメータ。v4.1-SmallをMeanFlowで蒸留した少ステップ生成版。条件エンコーダ・音声長予測器を固定し、DiTを学習。量子化とは別の変更 | MeanFlow 4 | 2.88 GiB | 1.44 GiB | MIT |
| [v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) | 約3.29Bパラメータ。DiTと参照エンコーダを拡大し、共通エンコーダをT5Gemma 2へ変更。音声長予測器は本体の学習後に別途学習 | RF 40 | 12.25 GiB | 6.12 GiB | Gemma Terms of Use |

重み用VRAMは、各公式リポジトリのsafetensorsパラメータ集計値に、FP32は4バイト、BF16は2バイトを掛けた概算です（1 GiB = 2³⁰バイト、2026-10-11確認）。**推論全体のVRAM消費量やピークの実測値ではありません。** 中間値、Attentionの計算用メモリ、CUDAの管理領域などが別途必要で、入力テキスト・参照音声・生成音声が長いほど増える場合があります。コーデックをGPUで動かす場合は、その重みと計算用メモリも必要です。本環境の既定ではコーデックをCPUで動かします。MFはステップ数を減らしますが、重みのVRAMは通常Smallとほぼ同じです。

ステップ既定はUIのNum Stepsを空欄、APIのnum_stepsを未指定にした場合です。RFはRectified Flow、MFはMeanFlowを指します。

選択時に参考になる公式評価・推論上の違い:

- **v4.1-Small**: v4-Smallのモデルカードで推奨される更新版で、このラッパーの既定です。同条件のJSUT評価でSentence Kana-CERはv4-Smallの3.49%から3.43%へ改善しています。[v4.1の更新内容・評価](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small#-benchmarks)
- **v4.1-Small-MF**: 4ステップのRFより読み精度・話者類似度が改善し、40ステップRFに近づくよう学習されています。JSUTのSentence Kana-CERはRF 4が6.50%、MF 4が3.78%、RF 40が3.43%。CFGは蒸留時に学習済みで、推論時のCFG値やSway Samplingは適用されません。[MeanFlowの評価・制約](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)
- **v4-Large**: 公開された内部自動評価では、v4-Smallよりcaptionへの追従スコアとCAM++話者類似度の平均値が高くなっています。一方、読み精度の評価ではv4.1-Smallが上回り、JSUTのSentence Kana-CERはSmall 3.43%、Large 3.67%です。用途ごとに結果が異なります。[Largeの評価](https://huggingface.co/Aratako/Irodori-TTS-v4-Large#-benchmarks)

CERは文字誤り率で、低いほど良い指標です。上記のJSUT値はFP32・5 seedの平均で、RFは40または4ステップ、text CFG 3.0、参照音声・captionなしの条件です。MFは蒸留済みのguidanceを使用します。モデルカード間でJoyo評価の版が異なるため、その数値は混在させて比較しません。公開された自動評価は、人が聴いた自然さや音質の順位を示すものではありません。

### 精度・量子化と本環境

RTX 2080（VRAM 8GB）向けの既定はモデルCUDA/FP32、コーデックCPU/FP32です。Composeの `IRODORI_MODEL_PRECISION`、`IRODORI_CODEC_DEVICE`、`IRODORI_CODEC_PRECISION` で変更できます。UIでもデバイス・精度を選択可能です。BF16非対応GPUではBF16を選択肢から除外します。

公式の量子化版は、[v4-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4-Small-Quantized)、[v4.1-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized)、[v4-Large-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized)です。それぞれ元モデルを学習後にtorchaoで量子化した版で、元モデルの条件入力・生成方式を引き継ぎます。MFの少ステップ化と、量子化によるメモリ削減は別の仕組みです。

weight-onlyは対応するLinear層の重みを量子化し、dynamicは実行時の中間値（activation）も量子化します。公式版ではプロジェクタ・AdaLN・音声長予測器などはBF16のままで、コーデックも量子化の対象外です。[上流の量子化説明](https://github.com/Aratako/Irodori-TTS#quantization)

量子化版の指定例:

```text
Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only
Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-dynamic
Aratako/Irodori-TTS-v4.1-Small-Quantized/int4-weight-only
Aratako/Irodori-TTS-v4.1-Small-Quantized/float8-weight-only
Aratako/Irodori-TTS-v4.1-Small-Quantized/float8-dynamic
```

| 量子化形式 | 公式版の重み・中間値 | 重み容量の参考値：Small（v4 / v4.1） | 重み容量の参考値：Large | GPU要件・位置づけ |
|---|---|---|---|---|
| INT8 weight-only | INT8重み / BF16計算（W8A16） | 0.85 GiB（872 MiB） | 3.58 GiB（3,662 MiB） | NVIDIA CUDA。公式の汎用INT8推奨形式 |
| INT8 dynamic | INT8重み / 対応Linear層の中間値も動的INT8（W8A8） | 0.85 GiB（872 MiB） | 3.58 GiB（3,662 MiB） | NVIDIA CUDA。実行時にも量子化するため、速度は環境に依存 |
| INT4 weight-only | INT4重み / BF16計算（W4A16、group size 128） | 0.79 GiB（813 MiB） | 2.75 GiB（2,818 MiB） | NVIDIA CC 8.0以上。公開variant中で最小のcheckpoint。RTX 2080では使用不可 |
| FP8 weight-only | FP8重み / BF16中間値 | 0.85 GiB（873 MiB） | 3.58 GiB（3,665 MiB） | NVIDIA CC 8.9以上。RTX 2080では使用不可 |
| FP8 dynamic | FP8重み / 対応Linear層の中間値も動的FP8 | 0.88 GiB（906 MiB） | 3.57 GiB（3,659 MiB） | NVIDIA CC 8.9以上。RTX 2080では使用不可 |

量子化表の容量は、上記の公式モデルカードに掲載されたcheckpointサイズをGiB換算した参考値です。保存形式や量子化メタデータも含むため、GPU上の重み容量と一致するとは限らず、推論全体のVRAM実測値でもありません。量子化されない層が残るため、全パラメータに単純に1バイト（INT8/FP8）や0.5バイト（INT4）を掛ける計算では過小評価します。本環境のFP32実行では、公式BF16構成より非量子化層の重み・中間値の容量が増えます。

不対応のINT4/FP8を選んだ場合は理由を表示し、APIは400を返します。選択肢への掲載はすべてのGPUでの動作保証ではありません。通常LargeをFP32でGPUに載せるには重みだけで約12.25 GiB必要なため、RTX 2080の8GBには収まりません。BF16の概算値はBF16対応GPU向けです。

上記のBF16は公式モデルカードの構成です。本環境のRTX 2080では、v4.1-SmallのINT8 weight-only / dynamicをFP32で実行し、音声生成と通常版へのモデル切替を確認済みです。このFP32実行は本環境での実験扱いです。検証時のINT8 dynamicは通常版より合成に時間がかかりました。Largeの音声生成は本環境では未検証です。

モデルと同梱tokenizerは初回使用時に取得し、マウントしたHugging Faceキャッシュを再利用します。量子化モデルは選択したvariantとtokenizerだけを取得します。

## REST API

`POST /api/v1/synthesize` はmultipart/form-dataを受け付けます。既存の単一 `ref_wav_path` リクエストをそのまま使用できます。

```bash
curl -X POST http://localhost:7860/api/v1/synthesize \
  -F text='こんにちは。私の名前はシブタニです。' \
  -F ref_wav_path=/app/reference_audio/220628_Shibutani-P.cut.wav \
  -F auto_split=true \
  -F seed=677
```

参照音声とcaptionは任意です。参照のみ、captionのみ、両方、text-onlyの合成に対応します。caption非対応checkpointへのcaption指定は400を返します。

```bash
curl -X POST http://localhost:7860/api/v1/synthesize \
  -F text='こんにちは。' \
  -F caption='落ち着いた自然な声' \
  -F ref_wav_paths=/app/reference_audio/ref_01.wav \
  -F ref_wav_paths=/app/reference_audio/ref_02.wav
```

複数参照は `ref_wav_paths` を繰り返して順序付きで指定します。単一の `ref_wav_path` との同時指定は400です。UIでは複数アップロードし、表示順を変更できます。参照長の上限はcheckpoint設定に従い、v4-Small系では合計120秒です。

| パラメータ | 既定 |
|---|---|
| text | 必須 |
| caption | 空欄 |
| ref_wav_path / ref_wav_paths | 未指定 |
| checkpoint | v4.1-Small |
| num_steps | モデル既定（RF: 40、MeanFlow: 4） |
| num_candidates | 1 |
| seed | ランダム |
| cfg_scale_text / cfg_scale_caption / cfg_scale_speaker | 3.0 / 4.0 / 5.0 |
| duration_scale | 1.0 |
| auto_split | false |
| max_chars_per_chunk / chunk_silence_ms | 200 / 150 |

明示的な `num_steps` は維持します。MeanFlowでは上流の仕様に従いCFGやRF sampler設定が無視されます。

成功レスポンスの構造とWAV取得URLは従来どおりです:

```json
{
  "service": "irodori-tts",
  "sample_rate": 48000,
  "used_seed": 677,
  "candidates": [
    {"index": 1, "path": "/app/api_outputs/sample_001.wav", "url": "/files/sample_001.wav"}
  ],
  "stage_timings": [["sample_rf", 0.42]],
  "total_to_decode": 1.20,
  "messages": []
}
```

| Method | Path | 用途 |
|---|---|---|
| GET | /api/v1/health | service名・既定checkpoint |
| POST | /api/v1/synthesize | 音声合成 |
| GET | /files/{name} | 生成WAV取得 |
| GET | /api/docs | Swagger UI |
| GET | /api/openapi.json | OpenAPIスキーマ |
| GET | / | Gradio UI |

不正入力・参照ファイル不在は400、CUDA OOMは507、その他の合成失敗は500を返します。

## 参照音声・長文

`reference_audio/` の実音声はGit追跡外で、コンテナ内の `/app/reference_audio` にread-onlyマウントします。APIの参照パスはこの配下に限定し、相対パスも受け付けます。生成WAVは `api_outputs/irodori-tts/` に保存します。旧VoiceDesignの出力ディレクトリは削除しません。

`auto_split=true` は句点・改行を優先して分割し、順次合成後に無音を挟んで連結します。`num_candidates=1` が必要です。全chunkで同じ参照音声・caption・seedを共有し、seed未指定なら最初のchunkで確定した値を再利用します。

CRLF・CR・連続改行・先頭末尾の改行を正規化し、分割しない場合は改行を空白に置換します。UI/APIの合成・モデル切替は共通の排他で制御します。

## テスト・v3への復旧

依存関係がインストール済みの環境で実行します:

```bash
PYTHONPATH=Irodori-TTS:. python3 -m pytest tests/
```

v3最終状態は注釈付きタグ `v3-final`（親リポジトリ `4124c1f`、上流 `5f103a5`）です。復旧時は別チェックアウトからタグとサブモジュールを取得し、その構成で起動します。v3イメージと既存モデルキャッシュは維持します。

## クレジット

- 元リポジトリ: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- コード・モデルのライセンスおよび利用条件は元リポジトリと各モデルカードを参照してください。
