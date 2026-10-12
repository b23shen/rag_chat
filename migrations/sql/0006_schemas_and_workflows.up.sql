-- 0006: two schemas, and the tables that run workflows.
--
--   platform  what any workflow uses: datasets, artifacts, workflows, workflow_runs, action_runs
--   rag       what the RAG app reads and writes: documents, promotions, chunks, ...
--
-- One database, so foreign keys still work across the two (rag.documents → platform.artifacts).
-- The app connects with search_path = rag, platform, public, so unqualified names keep working.
-- Needs CREATE on the database for the app role (scripts/init_db.sh grants it).

CREATE SCHEMA IF NOT EXISTS platform;
CREATE SCHEMA IF NOT EXISTS rag;

ALTER TABLE artifacts         SET SCHEMA platform;
ALTER TABLE documents         SET SCHEMA rag;
ALTER TABLE promotions        SET SCHEMA rag;
ALTER TABLE promotion_sources SET SCHEMA rag;
ALTER TABLE extractions       SET SCHEMA rag;
ALTER TABLE corpus_versions   SET SCHEMA rag;
ALTER TABLE chunks            SET SCHEMA rag;
ALTER TABLE embedding_cache   SET SCHEMA rag;


-- A named group of artifacts: the input of a workflow run.
CREATE TABLE platform.datasets (
    id           UUID        NOT NULL DEFAULT gen_random_uuid(),
    name         VARCHAR     NOT NULL,
    description  VARCHAR,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

CREATE TABLE platform.dataset_artifacts (
    dataset_id   UUID NOT NULL REFERENCES platform.datasets (id) ON DELETE CASCADE,
    artifact_id  UUID NOT NULL REFERENCES platform.artifacts (id),
    PRIMARY KEY (dataset_id, artifact_id)
);


-- A workflow's runbook, synced from workflows/<name>/runbook.yaml when the app starts.
CREATE TABLE platform.workflows (
    name          VARCHAR     NOT NULL,
    version       INTEGER     NOT NULL,
    description   VARCHAR,
    definition    JSONB       NOT NULL,           -- the runbook, as parsed
    runbook_hash  VARCHAR(64) NOT NULL,           -- sha256 of the file; changes when it's edited
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (name)
);


-- One run of a workflow. A queued row IS the job: a worker claims it with
-- SELECT … FOR UPDATE SKIP LOCKED, so two workers never take the same run,
-- and a server restart loses nothing.
CREATE TABLE platform.workflow_runs (
    id                UUID        NOT NULL DEFAULT gen_random_uuid(),
    workflow_name     VARCHAR     NOT NULL REFERENCES platform.workflows (name),
    workflow_version  INTEGER     NOT NULL,
    dataset_id        UUID        REFERENCES platform.datasets (id),
    params            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status            VARCHAR     NOT NULL DEFAULT 'queued',  -- queued | running | succeeded | failed
    output            JSONB,                                  -- the run report (GET /runs/{id}/content)
    error             VARCHAR,
    worker            VARCHAR,                                -- host:pid that claimed it
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    PRIMARY KEY (id)
);
-- The worker's "next job" query only ever looks at queued rows.
CREATE INDEX ix_workflow_runs_queued ON platform.workflow_runs (created_at) WHERE status = 'queued';


-- One action (step) of a run, in runbook order.
CREATE TABLE platform.action_runs (
    id           UUID        NOT NULL DEFAULT gen_random_uuid(),
    run_id       UUID        NOT NULL REFERENCES platform.workflow_runs (id) ON DELETE CASCADE,
    position     INTEGER     NOT NULL,
    name         VARCHAR     NOT NULL,
    status       VARCHAR     NOT NULL DEFAULT 'pending',  -- pending | running | succeeded | failed | skipped
    output       JSONB,                                   -- the action's report
    error        VARCHAR,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    PRIMARY KEY (id),
    UNIQUE (run_id, position)
);
CREATE INDEX ix_platform_action_runs_run_id ON platform.action_runs (run_id);
