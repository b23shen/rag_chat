import pytest

from promo_rag.search.segment import WORD_LIST, segment


def test_jieba_keeps_poya_words_whole():
    words = segment("即享券限寶雅線上買使用,單筆滿額贈回饋金", "jieba").split()
    for w in ("即享券", "寶雅線上買", "滿額贈", "回饋金"):
        assert w in words


def test_numbers_lose_thousands_commas():
    assert "5000" in segment("單筆最高可享5,000元回饋", "jieba").split()
    assert "5000" in segment("單筆最高可享5,000元回饋", "bigram").split()


def test_bigram_needs_no_dictionary():
    assert segment("回饋金 LINE Pay", "bigram") == "回饋 饋金 line pay"


def test_none_only_normalizes():
    assert segment("滿＄５,０００", "none") == "滿$5000"


def test_punctuation_never_becomes_a_word():
    assert all(
        any(ch.isalnum() for ch in w) for w in segment("【活動期間】:10/1~12/31!", "jieba").split()
    )


@pytest.mark.parametrize("line", WORD_LIST.read_text(encoding="utf-8").split())
def test_every_listed_word_survives_segmentation(line):
    assert line.lower() in segment(line, "jieba").split()
