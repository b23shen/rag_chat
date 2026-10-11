-- 0004: a cache of embeddings, keyed by the exact text sent to the model.
--
-- Rebuilding a corpus version (new chunk settings, a changed page) then only calls the
-- embedding API for text that actually changed. Question embeddings can be cached too, so
-- eval runs (and CI) don't need the API for questions they've seen before.

CREATE TABLE embedding_cache (
    model       VARCHAR     NOT NULL,
    dims        INTEGER     NOT NULL,
    input_hash  VARCHAR(64) NOT NULL,          -- sha256 of the exact input, task prefix included
    embedding   VECTOR      NOT NULL,          -- no fixed size, so several models/dims can share it
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model, dims, input_hash)
);
