-- 0002: a log of fact extraction (step 3), one row per document per prompt version.
--
-- Every attempt is recorded, including failures, so "why does this promotion have no
-- dates?" has an answer. Only `ok` outputs become promotions rows.

CREATE TABLE extractions (
    id              UUID        NOT NULL DEFAULT gen_random_uuid(),
    document_id     UUID        NOT NULL REFERENCES documents (id),
    schema_name     VARCHAR     NOT NULL,          -- PromotionFacts | StoreNoticeFacts
    method          VARCHAR     NOT NULL,          -- rules (code read a fixed format) | llm
    model           VARCHAR,                       -- NULL for rules
    prompt_version  VARCHAR     NOT NULL,          -- hash of prompt + schema; a change means re-extract
    status          VARCHAR     NOT NULL,          -- ok | needs_review (failed checks) | failed (API error)
    output          JSONB,                         -- the last output, valid or not
    errors          JSONB       NOT NULL DEFAULT '[]'::jsonb,  -- check failures, per attempt
    attempts        INTEGER     NOT NULL,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    UNIQUE (document_id, schema_name, prompt_version)  -- re-running the same version is a no-op
);
