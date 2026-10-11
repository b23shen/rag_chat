-- Undo 0003.
DROP INDEX uq_corpus_versions_fingerprint_inputs;
ALTER TABLE corpus_versions RENAME COLUMN inputs_hash TO artifact_set_hash;
