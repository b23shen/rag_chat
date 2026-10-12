-- Undo 0006: drop the workflow tables and move everything back to public.
DROP TABLE platform.action_runs;
DROP TABLE platform.workflow_runs;
DROP TABLE platform.workflows;
DROP TABLE platform.dataset_artifacts;
DROP TABLE platform.datasets;

ALTER TABLE rag.embedding_cache   SET SCHEMA public;
ALTER TABLE rag.chunks            SET SCHEMA public;
ALTER TABLE rag.corpus_versions   SET SCHEMA public;
ALTER TABLE rag.extractions       SET SCHEMA public;
ALTER TABLE rag.promotion_sources SET SCHEMA public;
ALTER TABLE rag.promotions        SET SCHEMA public;
ALTER TABLE rag.documents         SET SCHEMA public;
ALTER TABLE platform.artifacts    SET SCHEMA public;

DROP SCHEMA rag;
DROP SCHEMA platform;
