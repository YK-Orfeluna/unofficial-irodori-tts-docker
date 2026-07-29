#!/bin/sh
# HF Hub のオフライン設定を起動時に自動決定する。
#
# 背景:
#   HF_HUB_OFFLINE=1 は HF Hub への HEAD リクエストを完全に抑止する。NW 遮断環境では
#   これが無いと resolve_checkpoint_path() がリクエスト毎にリトライを繰り返し、
#   合成 1 件あたり実測 +23 秒のオーバーヘッドになる (実推論自体は 1 秒未満)。
#   一方 offline=1 のままキャッシュが空だと、起動時のモデル解決が必ず失敗する。
#   静的な設定ではこの 2 つを両立できないため、起動時に判定する。
#
# 方針:
#   まず offline で起動を試す。キャッシュ不足による起動失敗 (uvicorn の exit 3) の
#   場合のみ、1 度だけ online で起動し直してキャッシュを温める。次回起動以降は
#   キャッシュが揃っているので offline で立ち上がる。
#   HF_HUB_OFFLINE が明示指定されていれば自動判定せずその値に従う。

set -u

# $1 に offline フラグ値、以降にアプリの引数を取る。
start() {
    offline=$1
    shift
    HF_HUB_OFFLINE=$offline TRANSFORMERS_OFFLINE=$offline \
        uv run --no-sync python "$@" &
    child=$!
    # PID 1 の sh は SIGTERM を子プロセスへ自動転送しないため明示的に中継する。
    # これが無いと docker compose down が 10 秒の SIGKILL 待ちになる。
    trap 'kill -TERM "$child" 2>/dev/null' TERM INT
    wait "$child"
    status=$?
    trap - TERM INT
    # シグナルで wait が抜けた場合、子の終了を待ち直す。
    if [ "$status" -gt 128 ]; then
        wait "$child" 2>/dev/null || true
    fi
    return "$status"
}

if [ -n "${HF_HUB_OFFLINE:-}" ]; then
    echo "[entrypoint] HF_HUB_OFFLINE=${HF_HUB_OFFLINE} は明示指定。自動判定をスキップする" >&2
    exec uv run --no-sync python "$@"
fi

echo "[entrypoint] HF_HUB_OFFLINE=1 (キャッシュのみ) で起動を試行する" >&2
# `if start ...; then` の直後の $? は if 文自体の終了状態 (条件が偽なら 0) になり
# start の終了コードを取り落とすため、必ず直接呼んでから $? を取る。
start 1 "$@"
status=$?

if [ "$status" -eq 0 ]; then
    exit 0
fi

# 3 = uvicorn の startup failure。キャッシュ不足はここに該当する。
# 2 (precision 設定不正) や 143 (SIGTERM) では再試行しない。
if [ "$status" -ne 3 ]; then
    exit "$status"
fi

echo "[entrypoint] offline 起動が失敗 (exit 3)。HF_HUB_OFFLINE=0 で再試行しキャッシュを取得する" >&2
start 0 "$@"
