-- 0005: at most one active corpus version.
--
-- Search reads the active version. Two active versions would make results depend on which
-- one a query happened to pick, so the database itself refuses a second one.

CREATE UNIQUE INDEX uq_corpus_versions_one_active ON corpus_versions (status) WHERE status = 'active';
