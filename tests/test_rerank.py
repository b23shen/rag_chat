"""The Cohere rerank client, against a fake server (no network)."""

import json

import httpx
import pytest

from promo_rag.config import get_settings
from promo_rag.search.rerank import RerankError, cohere_rerank, scores_in_input_order


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _server(*responses):
    """A fake Cohere that answers with these (status, body) pairs in order."""
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status, body = responses[len(requests) - 1]
        return httpx.Response(status, json=body, headers={"retry-after": "0"})

    return httpx.MockTransport(handle), requests


def test_scores_come_back_in_input_order():
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.5},
        ]
    }
    assert scores_in_input_order(payload, 3) == [0.1, 0.5, 0.9]


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.9}],  # a score is missing
        [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}],  # repeated
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 5, "relevance_score": 0.8},
        ],  # out of range
    ],
)
def test_bad_results_are_rejected(results):
    with pytest.raises(RerankError):
        scores_in_input_order({"results": results}, 2)


def test_request_has_model_query_documents_and_key():
    ok = {"results": [{"index": 0, "relevance_score": 0.7}]}
    transport, requests = _server((200, ok))
    call = cohere_rerank("rerank-v4.0-fast", transport=transport)
    assert call("可以帶狗嗎", ["寵物入店規定"]) == [0.7]
    body = json.loads(requests[0].content)
    assert body == {
        "model": "rerank-v4.0-fast",
        "query": "可以帶狗嗎",
        "documents": ["寵物入店規定"],
    }
    assert requests[0].headers["authorization"] == "Bearer test-key"


def test_rate_limit_is_retried():
    ok = {"results": [{"index": 0, "relevance_score": 0.7}]}
    transport, requests = _server((429, {"message": "slow down"}), (200, ok))
    assert cohere_rerank(transport=transport)("q", ["d"]) == [0.7]
    assert len(requests) == 2


def test_other_errors_are_not_retried():
    transport, requests = _server((400, {"message": "invalid model"}))
    with pytest.raises(RerankError, match="HTTP 400"):
        cohere_rerank(transport=transport)("q", ["d"])
    assert len(requests) == 1


def test_no_documents_means_no_call():
    transport, requests = _server()
    assert cohere_rerank(transport=transport)("q", []) == []
    assert not requests


def test_missing_key_says_where_to_set_it(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "")  # overrides .env, like an empty line there
    get_settings.cache_clear()
    with pytest.raises(RerankError, match="COHERE_API_KEY"):
        cohere_rerank()
