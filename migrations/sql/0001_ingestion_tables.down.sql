-- Undo 0001. Drop in reverse dependency order; indexes go with their tables.
-- The vector extension stays: other databases or migrations may use it.
DROP TABLE chunks;
DROP TABLE corpus_versions;
DROP TABLE promotion_sources;
DROP TABLE promotions;
DROP TABLE documents;
DROP TABLE artifacts;
