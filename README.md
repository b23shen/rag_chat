# Retail Promotions RAG Assistant

A chat assistant that answers questions about a retailer's promotions, offers and products, in Chinese and English, from the retailer's public web pages. An LLM agent routes each question to SQL tools (count and list promotions by date, store or payment method) or to hybrid search (pgvector + full-text, rank fusion, reranking) for rules and fine print. Every answer cites its sources, and every design choice is measured on a golden question set.

*Work in progress.* Done: **M1, ingestion as a workflow** ([handoff](docs/handoffs/M1_ingestion.md)). Next: M2, search and eval.

## Layout

```
src/promo_rag/
  platform/            the workflow engine, shared by every workflow
    api.py             HTTP API: datasets, artifacts, start a workflow, run status and content
    runbook.py         runbooks (workflows/<name>/runbook.yaml), synced to platform.workflows
    runner.py          the run queue (FOR UPDATE SKIP LOCKED) and the action runner
    worker.py          takes queued runs and executes them
    datasets.py        datasets and artifacts
  agents/              LLM agents, shared by every workflow
    invoke.py          invoke_agent(name, messages, variables=, response_schema=, trace=) → AgentResult
    config.py          AgentConfig + a registry that finds each workflow's agents.py
    prompts.py         Markdown prompts with {{variable}} slots; version = hash of the text
    models.py          model roles (chat / helper / judge) from .env; retries set here only
    tools/ middleware/ shared tools and middleware, referenced by name
    tracing.py         Langfuse (off without keys)
  search/              shared by ingestion and search: word splitting (jieba), embeddings + cache
  workflows/
    ingestion/         runbook.yaml, actions/, agents.py, prompts/, sources/, handler.py (CLI)
  db/  storage/        tables (schemas platform and rag) and the blob store
```

## Setup (macOS)

Prerequisites: PostgreSQL 18 from the EDB installer, the Xcode Command Line Tools, [uv](https://docs.astral.sh/uv/), and a Gemini API key.

```bash
cp .env.example .env                 # then set GEMINI_API_KEY in .env
bash scripts/install_pgvector.sh     # builds pgvector; sudo asks for your Mac password
bash scripts/init_db.sh              # role, database, extension, grants; asks for the postgres password
uv sync                              # Python 3.12 + dependencies
uv run python scripts/smoke_test.py  # checks Postgres, pgvector, embeddings and structured output
uv run alembic upgrade head          # creates the tables
```

`init_db.sh` generates the app's database password and writes it to `.env`. `.env` is gitignored.

## Ingestion as a workflow

A dataset of fetched pages goes in; an active, searchable corpus version comes out. The steps are listed in [`runbook.yaml`](src/promo_rag/workflows/ingestion/runbook.yaml): parse → extract → chunk → embed → activate.

```bash
uv run serve                         # the API on http://127.0.0.1:8000 (interactive docs at /docs)
uv run worker                        # in another terminal: runs queued workflow runs
```

```bash
# 1. A dataset of pages: from a snapshot folder...
uv run ingest import-snapshot <snapshot folder>          # prints the dataset id
#    ...or over HTTP, one page at a time
curl -X POST localhost:8000/datasets -H 'content-type: application/json' -d '{"name": "re-crawl"}'
curl -X POST localhost:8000/datasets/<id>/artifacts -F file=@page.html -F source_url=https://…

# 2. Start the workflow (202: queued), then follow it
curl -X POST localhost:8000/workflows/ingestion/start -H 'content-type: application/json' \
     -d '{"dataset_id": "<id>", "params": {"segmenter": "jieba"}}'
curl localhost:8000/runs/<run_id>/status         # queued | running | succeeded | failed, per action
curl localhost:8000/runs/<run_id>/content        # each action's report, once finished
```

Runs are rows in `platform.workflow_runs`; a worker claims the oldest queued one with `SELECT … FOR UPDATE SKIP LOCKED`, so several workers never take the same run and a restart loses nothing. Each action's status, timing, output and error is recorded in `platform.action_runs`.

## Running one step at a time

The same actions, as CLI commands (handy while developing):

```bash
uv run ingest parse        # artifacts → documents (clean text, doc type from the site category, content hash)
uv run ingest extract      # documents → promotions (rules for fixed formats, Gemini structured output otherwise)
                           #   --only <url part> --force to redo one page
uv run ingest chunk        # documents → chunks of a corpus version (header line, one chunk per FAQ question,
                           #   jieba keyword text). Options: --segmenter none|jieba|bigram, --chunk-tokens, --no-header
uv run ingest embed        # chunk embeddings (gemini-embedding-2, 1536 dims), cached by the exact input text
uv run ingest activate     # check the version (every chunk embedded, nothing stale), make it the one search reads
```

Every step is safe to re-run. Parse skips pages that aren't HTTP 200, have no text (image-only), or duplicate another page, and records the reason in `documents.skip_reason`. Extract checks every output in code (dates must be quoted from the page, start ≤ end, numbers must appear on the page), retries once with the errors, and logs each run in `extractions`. Only outputs that pass become `promotions` rows.

A snapshot folder holds `sources.csv` and `raw/<slug>.html`. Raw pages are never committed; [`data/sources.csv`](data/sources.csv) lists each page's URL and sha256 instead.
