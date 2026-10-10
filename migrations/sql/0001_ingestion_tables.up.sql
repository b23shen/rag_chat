-- 0001: ingestion tables.
--
--   artifacts 1──< documents >──< promotions      (via promotion_sources)
--                      │
--                      └──< chunks >── corpus_versions
--
-- Postgres holds all metadata; the blob store holds only raw bytes.
-- Keep this in sync with src/promo_rag/db/models.py (`uv run alembic check` compares them).

-- pgvector must already be installed on the server; scripts/init_db.sh enables it.
CREATE EXTENSION IF NOT EXISTS vector;


-- One fetch: the exact bytes one URL returned at one time.
CREATE TABLE artifacts (
    id           UUID        NOT NULL DEFAULT gen_random_uuid(),
    sha256       VARCHAR(64) NOT NULL,
    source_url   VARCHAR     NOT NULL,
    final_url    VARCHAR,                          -- after redirects
    fetched_at   TIMESTAMPTZ NOT NULL,
    http_status  INTEGER     NOT NULL,             -- 404s are kept: a re-crawl 404 means "page removed"
    mime_type    VARCHAR,
    byte_size    INTEGER     NOT NULL,
    blob_key     VARCHAR     NOT NULL,             -- raw/<sha256[:2]>/<sha256>.html; same bytes, one blob
    fetch_meta   JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- what the crawler knew: slug, category, list date
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    UNIQUE (source_url, sha256)                    -- re-importing the same fetch is a no-op
);
CREATE INDEX ix_artifacts_sha256 ON artifacts (sha256);


-- One parsed version of a page.
CREATE TABLE documents (
    id                   UUID        NOT NULL DEFAULT gen_random_uuid(),
    artifact_id          UUID        NOT NULL REFERENCES artifacts (id),
    source_url           VARCHAR     NOT NULL,
    title                VARCHAR     NOT NULL,
    language             VARCHAR     NOT NULL DEFAULT 'zh',
    doc_type             VARCHAR     NOT NULL,     -- promotion | store_notice | announcement | faq_policy | catalog
    category             VARCHAR,                  -- the site's own category, e.g. 主題活動
    published_date       DATE,
    content_hash         VARCHAR(64) NOT NULL,     -- sha256 of the cleaned text; finds duplicate pages
    full_text            VARCHAR     NOT NULL,
    image_urls           VARCHAR[]   NOT NULL DEFAULT '{}',  -- kept for a later vision step
    version              INTEGER     NOT NULL DEFAULT 1,
    is_latest            BOOLEAN     NOT NULL DEFAULT true,
    previous_version_id  UUID        REFERENCES documents (id),
    status               VARCHAR     NOT NULL DEFAULT 'active',  -- active | archived | skipped
    skip_reason          VARCHAR,                  -- e.g. "http 404", "duplicate of <id>"
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);
CREATE INDEX ix_documents_content_hash ON documents (content_hash);
-- Only one latest version per URL.
CREATE UNIQUE INDEX uq_documents_latest_url ON documents (source_url) WHERE is_latest;


-- Structured facts for one promotion. The SQL tools (count/list promotions) query this table.
CREATE TABLE promotions (
    id                UUID        NOT NULL DEFAULT gen_random_uuid(),
    dedup_key         VARCHAR     NOT NULL,        -- normalized title + dates; same campaign on two pages = one row
    title             VARCHAR     NOT NULL,
    promo_type        VARCHAR     NOT NULL,        -- discount | gift | points | coupon | card_offer | store_opening | clearance_sale | ...
    start_date        DATE,
    end_date          DATE,                        -- NULL = open-ended
    stores            VARCHAR[]   NOT NULL DEFAULT '{}',
    channels          VARCHAR[]   NOT NULL DEFAULT '{}',
    payment_methods   VARCHAR[]   NOT NULL DEFAULT '{}',
    products          VARCHAR[]   NOT NULL DEFAULT '{}',
    min_spend         NUMERIC,
    discount_text     VARCHAR     NOT NULL,        -- the rule as written on the page
    extraction_model  VARCHAR     NOT NULL,
    extracted_at      TIMESTAMPTZ NOT NULL,
    raw               JSONB       NOT NULL,        -- the full Pydantic output
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    UNIQUE (dedup_key)
);


-- Which pages describe a promotion (usually one, sometimes two).
CREATE TABLE promotion_sources (
    promotion_id  UUID NOT NULL REFERENCES promotions (id) ON DELETE CASCADE,
    document_id   UUID NOT NULL REFERENCES documents (id),
    PRIMARY KEY (promotion_id, document_id)
);


-- One index build: these pages, indexed with these settings.
CREATE TABLE corpus_versions (
    id                 UUID        NOT NULL DEFAULT gen_random_uuid(),
    fingerprint        VARCHAR(16) NOT NULL,       -- hash of config; if it changes, chunks must be rebuilt
    config             JSONB       NOT NULL,       -- chunk size, overlap, embed model, dims, segmenter, ...
    artifact_set_hash  VARCHAR(64),                -- hash of the sorted artifact hashes in this build
    n_documents        INTEGER     NOT NULL DEFAULT 0,
    n_chunks           INTEGER     NOT NULL DEFAULT 0,
    status             VARCHAR     NOT NULL DEFAULT 'building',  -- building | active | retired
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);
CREATE INDEX ix_corpus_versions_fingerprint ON corpus_versions (fingerprint);


-- The search units.
CREATE TABLE chunks (
    id                 UUID        NOT NULL DEFAULT gen_random_uuid(),
    document_id        UUID        NOT NULL REFERENCES documents (id),
    corpus_version_id  UUID        NOT NULL REFERENCES corpus_versions (id),
    chunk_index        INTEGER     NOT NULL,       -- position within the page: 0, 1, 2, ...
    text               VARCHAR     NOT NULL,       -- what the LLM reads (header line + body)
    search_text        VARCHAR     NOT NULL,       -- jieba-segmented text for keyword search
    tsv                TSVECTOR    NOT NULL GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED,
    embedding          VECTOR(1536),               -- NULL until the embed step has run
    embed_model        VARCHAR,                    -- vectors from different models can't be compared
    token_count        INTEGER     NOT NULL,
    -- Copied from the document/promotion so filters run on this table alone.
    doc_type           VARCHAR     NOT NULL,
    language           VARCHAR     NOT NULL,
    start_date         DATE,
    end_date           DATE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    UNIQUE (document_id, corpus_version_id, chunk_index)
);
CREATE INDEX ix_chunks_corpus_version_id ON chunks (corpus_version_id);
-- Vector search: approximate nearest neighbours by cosine distance (the <=> operator).
CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
-- Keyword search: tsv @@ query.
CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv);
