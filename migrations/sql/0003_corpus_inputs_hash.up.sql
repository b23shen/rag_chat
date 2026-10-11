-- 0003: a corpus version is identified by its settings (fingerprint) and its inputs.
--
-- Chunks are built from each document's text AND the extracted facts in its header line
-- (title | dates | stores), so the old artifact_set_hash (pages only) missed changes to
-- the facts. inputs_hash covers both. Same fingerprint + same inputs_hash = same chunks.

ALTER TABLE corpus_versions RENAME COLUMN artifact_set_hash TO inputs_hash;
CREATE UNIQUE INDEX uq_corpus_versions_fingerprint_inputs ON corpus_versions (fingerprint, inputs_hash);
