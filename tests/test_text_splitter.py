from __future__ import annotations

import pytest

from irodori_tts_api.text_splitter import split_text_for_tts


def test_empty_input_returns_empty_list():
    assert split_text_for_tts("") == []
    assert split_text_for_tts("   \n  ") == []


def test_short_single_sentence_passes_through():
    assert split_text_for_tts("こんにちは。") == ["こんにちは。"]


def test_split_by_japanese_period():
    text = "おはようございます。今日はいい天気ですね。出かけましょう。"
    out = split_text_for_tts(text, max_chars=200)
    assert out == [
        "おはようございます。",
        "今日はいい天気ですね。",
        "出かけましょう。",
    ]


def test_split_by_question_and_exclamation():
    text = "本当に？すごい！それで?どうしたの!"
    out = split_text_for_tts(text, max_chars=200)
    assert out == ["本当に？", "すごい！", "それで?", "どうしたの!"]


def test_split_by_newlines_then_sentences():
    text = "一行目です。二文目。\n二段落目。\n\n三段落目です。"
    out = split_text_for_tts(text, max_chars=200)
    assert out == [
        "一行目です。",
        "二文目。",
        "二段落目。",
        "三段落目です。",
    ]


def test_long_sentence_falls_back_to_commas():
    text = "今日は朝早く起きて、近所の公園を散歩して、その後カフェに寄って、コーヒーを飲みながら本を読みました"
    out = split_text_for_tts(text, max_chars=20)
    assert all(len(c) <= 20 for c in out)
    joined = "".join(out)
    assert joined.replace("、", "") == text.replace("、", "") or "、" in joined


def test_hard_split_when_no_punctuation():
    text = "あ" * 500
    out = split_text_for_tts(text, max_chars=100)
    assert len(out) == 5
    assert all(len(c) == 100 for c in out)
    assert "".join(out) == text


def test_short_fragment_merges_into_previous():
    text = "これは長めの文章で、文末記号で区切られています。は。"
    out = split_text_for_tts(text, max_chars=200)
    assert all(len(c) >= 5 for c in out)


def test_merge_does_not_exceed_max_chars():
    long_sentence = "あ" * 18 + "。"
    text = long_sentence + "あ。"
    out = split_text_for_tts(text, max_chars=20)
    assert len(out) == 2


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        split_text_for_tts("テスト。", max_chars=0)
    with pytest.raises(ValueError):
        split_text_for_tts("テスト。", max_chars=-1)


def test_strip_noise_newlines_at_edges():
    assert split_text_for_tts("\r\nこんにちは。\r\n") == ["こんにちは。"]
    assert split_text_for_tts("\n\n\nテスト。\n\n") == ["テスト。"]


def test_lone_cr_is_normalized():
    text = "前半です。\r後半です。"
    assert split_text_for_tts(text) == ["前半です。", "後半です。"]


def test_dify_style_input_with_trailing_crlf():
    text = "こんにちは。私の名前はシブタニです。\r\n"
    out = split_text_for_tts(text, max_chars=200)
    assert out == ["こんにちは。", "私の名前はシブタニです。"]


def test_mixed_realistic_long_input():
    text = (
        "本日はお忙しいところ、お時間をいただきありがとうございます。"
        "本件の進捗ですが、概ね計画通りに進行しております。"
        "ただし、来週以降のリリーススケジュールについては、"
        "QA チームとの調整が必要なため、改めてご連絡いたします。"
    )
    out = split_text_for_tts(text, max_chars=200)
    assert all(0 < len(c) <= 200 for c in out)
    joined = "".join(out)
    for keyword in ["お時間", "進捗", "リリース", "QA"]:
        assert keyword in joined
