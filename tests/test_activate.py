from promo_rag.workflows.ingestion.actions.activate import VersionStats, problems


def _stats(**overrides):
    base = {
        "n_chunks": 232,
        "missing_embeddings": 0,
        "wrong_model": 0,
        "wrong_dims": 0,
        "docs_without_chunks": [],
        "stale": False,
    }
    return VersionStats(**(base | overrides))


def test_a_complete_version_has_no_problems():
    assert problems(_stats()) == []


def test_each_gap_blocks_activation_with_a_reason():
    assert "no embedding" in problems(_stats(missing_embeddings=3))[0]
    assert "different model" in problems(_stats(wrong_model=1))[0]
    assert "wrong number of dims" in problems(_stats(wrong_dims=1))[0]
    assert "have no chunks" in problems(_stats(docs_without_chunks=["https://x/1"]))[0]
    assert "changed after this version was built" in problems(_stats(stale=True))[0]
    assert "no chunks" in problems(_stats(n_chunks=0))[0]
