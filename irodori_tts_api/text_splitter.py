"""TTS 長文合成のためのテキスト分割ユーティリティ。

Diffusion ベース TTS は一度に長文を投げると後半が崩れやすいため、
事前に句読点で分割して chunk 単位で合成 → 連結する方針を採る。
"""

from __future__ import annotations

import re

# 句点(文末)候補。読み上げ時に明確な区切りを生む文字。
_SENTENCE_END_RE = re.compile(r"(?<=[。．！？!?])")
# 読点。最終手段の手前で使う「やや弱い区切り」。
_CLAUSE_SEP_RE = re.compile(r"(?<=[、,])")

# これ未満の長さの chunk は前の chunk に併合する(「は。」のような句点取り残し対策)。
# 「すごい！」など正当な短文を巻き込まないよう、句点 1 文字相当の極小値に絞る。
_MERGE_THRESHOLD = 3


def split_text_for_tts(text: str, max_chars: int = 200) -> list[str]:
    """テキストを TTS 合成用の chunk 列に分割する。

    優先順:
      1. 改行で粗分割
      2. 句点(。．！？!?)で文単位に
      3. それでも max_chars を超える文は読点(、,)で再分割
      4. それでも超える場合は max_chars で強制カット
    そして短すぎる断片は前の chunk に併合する。

    Args:
        text: 入力テキスト。
        max_chars: 1 chunk の最大文字数。

    Returns:
        chunk 文字列のリスト。空入力なら空リスト。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be >= 1.")
    if text is None:
        return []

    # クライアント側(multipart parser 等)で混入するノイズ改行(先頭末尾・連続改行・CR 単独)を除去し、
    # 「文字と文字に挟まれた改行」だけを分割キーとして残す。
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]*\n[ \t]*", "\n", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    normalized = normalized.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in _split_by_pattern(line, _SENTENCE_END_RE):
            for piece in _further_split_if_needed(sentence, max_chars):
                chunks.append(piece)

    return _merge_short_tail(chunks, max_chars)


def _further_split_if_needed(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    for clause in _split_by_pattern(sentence, _CLAUSE_SEP_RE):
        if len(clause) <= max_chars:
            parts.append(clause)
        else:
            parts.extend(_hard_split(clause, max_chars))
    return parts


def _split_by_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    """正規表現の lookbehind で分割し、空文字を除いて返す。"""
    return [s.strip() for s in pattern.split(text) if s.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _merge_short_tail(chunks: list[str], max_chars: int) -> list[str]:
    """先頭から走査し、極端に短い chunk を直前 chunk に併合する。

    併合後の長さが max_chars を超えるなら併合しない(過長回避)。
    """
    if not chunks:
        return chunks
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) < _MERGE_THRESHOLD and len(merged[-1]) + len(chunk) <= max_chars:
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)
    return merged
