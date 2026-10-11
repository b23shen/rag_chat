"""Embedding with a fake API and an in-memory cache (no network, no database)."""

import pytest

from promo_rag.search.embed import EmbedError, document_input, embed_inputs, query_input
from promo_rag.workflows.ingestion.index_config import IndexConfig

DIMS = 4


class MemoryCache:
    def __init__(self):
        self.rows = {}

    def get_many(self, model, dims, hashes):
        return {h: self.rows[(model, dims, h)] for h in hashes if (model, dims, h) in self.rows}

    def put_many(self, model, dims, vectors):
        self.rows.update({(model, dims, h): v for h, v in vectors.items()})


class FakeApi:
    """Returns a recognizable vector per text: [len(text), 0, 0, 1]."""

    def __init__(self, merge=False, dims=DIMS):
        self.calls, self.merge, self.dims = [], merge, dims

    def __call__(self, texts):
        self.calls.append(list(texts))
        vectors = [[float(len(t))] + [0.0] * (self.dims - 2) + [1.0] for t in texts]
        return vectors[:1] if self.merge else vectors  # merge: the list[str] bug


def _embed(inputs, cache, api, batch_size=50):
    return embed_inputs(inputs, model="m", dims=DIMS, cache=cache, call=api, batch_size=batch_size)


def test_vectors_come_back_in_input_order_and_go_into_the_cache():
    cache, api = MemoryCache(), FakeApi()
    result = _embed(["a", "bbb", "cc"], cache, api)
    assert [v[0] for v in result.vectors] == [1.0, 3.0, 2.0]
    assert (result.from_api, result.from_cache, result.api_calls) == (3, 0, 1)
    assert len(cache.rows) == 3


def test_second_run_uses_the_cache_and_only_sends_new_text():
    cache, api = MemoryCache(), FakeApi()
    _embed(["a", "bbb"], cache, api)
    result = _embed(["a", "bbb", "dddd"], cache, api)
    assert api.calls[-1] == ["dddd"]
    assert (result.from_cache, result.from_api) == (2, 1)


def test_identical_texts_are_sent_once():
    api = FakeApi()
    result = _embed(["same", "same", "other"], MemoryCache(), api)
    assert api.calls == [["same", "other"]]
    assert result.vectors[0] == result.vectors[1]


def test_batches_respect_the_batch_size():
    api = FakeApi()
    _embed([f"text {i}" for i in range(5)], MemoryCache(), api, batch_size=2)
    assert [len(c) for c in api.calls] == [2, 2, 1]


def test_fewer_vectors_than_texts_is_an_error():
    """The gemini-embedding-2 gotcha: a plain list[str] came back as ONE blended vector."""
    with pytest.raises(EmbedError, match="sent 3 texts, got 1"):
        _embed(["a", "b", "c"], MemoryCache(), FakeApi(merge=True))


def test_wrong_dimensions_is_an_error():
    with pytest.raises(EmbedError, match="expected 4 dims"):
        _embed(["a"], MemoryCache(), FakeApi(dims=8))


def test_input_formats():
    cfg = IndexConfig(embed_model="m", embed_dims=DIMS, word_list="x")
    assert document_input(cfg.embed_doc_format, "十分賺", "內文") == "title: 十分賺 | text: 內文"
    assert (
        query_input(cfg.embed_query_format, "週五優惠?") == "task: search result | query: 週五優惠?"
    )
