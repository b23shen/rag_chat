import itertools
from datetime import date

from promo_rag.workflows.ingestion.chunking import (
    HeaderFacts,
    Unit,
    chunk_document,
    estimate_tokens,
    header_line,
    pack,
    split_sentences,
)
from promo_rag.workflows.ingestion.index_config import IndexConfig

CFG = IndexConfig(embed_model="test", embed_dims=8, word_list="test")


def test_split_sentences_on_chinese_and_ascii_marks_and_line_ends():
    units = split_sentences("第一句。第二句!第三句\n第四句;")
    assert [u.text for u in units] == ["第一句。", "第二句!", "第三句", "第四句;"]
    assert [u.sep for u in units] == ["", "", "\n", "\n"]


def test_pack_keeps_the_budget_and_repeats_the_tail():
    units = [Unit("字" * 44, "\n") for _ in range(10)]  # ~40 tokens each
    bodies = pack(units, budget=100, overlap=45, cfg=CFG)
    assert len(bodies) > 1
    assert all(estimate_tokens(b, CFG) <= 100 for b in bodies)
    # each body after the first starts with the last unit of the one before
    for prev, nxt in itertools.pairwise(bodies):
        assert nxt.split("\n")[0] == prev.split("\n")[-1]


def test_a_run_on_paragraph_is_cut_to_fit():
    bodies = pack([Unit("字" * 2000, "\n")], budget=100, overlap=0, cfg=CFG)
    assert len(bodies) > 1 and all(estimate_tokens(b, CFG) <= 100 for b in bodies)


def test_header_line():
    campaign = HeaderFacts("十分賺", date(2026, 10, 7), date(2026, 11, 3))
    assert header_line(campaign) == "十分賺 | 2026-10-07 ~ 2026-11-03"
    opening = HeaderFacts("【新店開幕】鼓山大順店", date(2026, 8, 22), None, ("鼓山大順店",), True)
    assert header_line(opening) == "【新店開幕】鼓山大順店 | 2026-08-22"  # store not repeated
    limited = HeaderFacts("限店活動", date(2026, 10, 7), None, ("台中大墩店", "台南和緯店"))
    assert header_line(limited) == "限店活動 | 2026-10-07 ~ | 台中大墩店, 台南和緯店"
    assert header_line(HeaderFacts("會員須知")) == "會員須知"


def test_faq_gets_one_chunk_per_question_with_the_question_in_the_header():
    text = "Q: 寵物入店規定\nA: 請將寵物放進寵物袋\n\nQ: 停車資訊\nA: 請致電門市"
    drafts = chunk_document(text, HeaderFacts("門市相關-常見問題"), CFG)
    assert [d.text.split("\n")[0] for d in drafts] == [
        "門市相關-常見問題 | 寵物入店規定",
        "門市相關-常見問題 | 停車資訊",
    ]


def test_every_chunk_starts_with_the_header_and_fits():
    text = "\n".join(f"第{i}條規定,單筆消費滿$800享88折優惠,每人限用一次。" for i in range(80))
    drafts = chunk_document(text, HeaderFacts("活動", date(2026, 10, 1), date(2026, 12, 31)), CFG)
    assert len(drafts) > 1
    assert all(d.text.startswith("活動 | 2026-10-01 ~ 2026-12-31\n") for d in drafts)
    assert all(d.token_count <= CFG.chunk_tokens for d in drafts)


def test_fingerprint_changes_with_any_setting():
    assert (
        CFG.fingerprint()
        == IndexConfig(embed_model="test", embed_dims=8, word_list="test").fingerprint()
    )
    assert CFG.fingerprint() != CFG.model_copy(update={"segmenter": "bigram"}).fingerprint()
    assert CFG.fingerprint() != CFG.model_copy(update={"chunk_tokens": 256}).fingerprint()
