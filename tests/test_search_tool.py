"""search_promotion_text, with a fake search (no database, no API calls)."""

import uuid

import pytest
from pydantic import ValidationError

from promo_rag.agents.tools import TOOLS, search_text
from promo_rag.agents.tools.search_text import SearchArgs, search_promotion_text
from promo_rag.search.retrieve import Hit, SearchError


def _hit(rank: int, score: float, reranked: bool = True) -> Hit:
    return Hit(
        rank=rank,
        chunk_id=uuid.uuid4(),
        source_url=f"https://www.poya.com.tw/faq/{rank}",
        title="門市相關-常見問題",
        text="Q: 可以帶寵物嗎?\n導盲犬以外的寵物請勿入店。",
        score=score,
        reranked=reranked,
    )


def _run(monkeypatch, hits=None, error=None) -> dict:
    def fake_search(session, query, *, k):
        if error:
            raise error
        return hits[:k]

    monkeypatch.setattr(search_text, "search", fake_search)
    return search_promotion_text(None, SearchArgs(query="可以帶狗嗎?", k=2))


def test_passages_come_with_their_pages_and_no_note(monkeypatch):
    result = _run(monkeypatch, [_hit(1, 0.9081), _hit(2, 0.5)])
    assert [r["rank"] for r in result["results"]] == [1, 2]
    assert result["results"][0]["score"] == 0.908
    assert result["results"][0]["url"].endswith("/faq/1")
    assert "note" not in result


def test_a_low_top_score_says_the_pages_may_not_answer(monkeypatch):
    result = _run(monkeypatch, [_hit(1, 0.42)])
    assert "Low relevance" in result["note"] and "0.42" in result["note"]


def test_without_the_reranker_scores_mean_nothing(monkeypatch):
    result = _run(monkeypatch, [_hit(1, 0.42, reranked=False)])
    assert "reranker was unavailable" in result["note"]


def test_no_passages(monkeypatch):
    assert _run(monkeypatch, [])["note"] == "No passages found."


def test_search_errors_go_back_to_the_model(monkeypatch):
    result = _run(monkeypatch, error=SearchError("no active corpus version"))
    assert "no active corpus version" in result["error"]


def test_bad_arguments_are_rejected():
    with pytest.raises(ValidationError):
        SearchArgs(query="退貨", k=50)
    with pytest.raises(ValidationError):
        SearchArgs(query="")


def test_the_tool_is_registered():
    assert "search_promotion_text" in TOOLS
