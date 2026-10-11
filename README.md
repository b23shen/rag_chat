# Retail Promotions RAG Assistant

A chat assistant that answers questions about a retailer's promotions, offers and products, in Chinese and English, from the retailer's public web pages. An LLM agent routes each question to SQL tools (count and list promotions by date, store or payment method) or to hybrid search (pgvector + full-text, rank fusion, reranking) for rules and fine print. Every answer cites its sources, and every design choice is measured on a golden question set.

*Work in progress.*

## Layout

```
src/promo_rag/
  agents/              shared by every workflow
    invoke.py          invoke_agent(name, messages, variables=, response_schema=, context=) → AgentResult
    config.py          AgentConfig + a registry that finds each workflow's agents.py
    prompts.py         Markdown prompts with {{variable}} slots; version = hash of the text
    models.py          model roles (chat / helper / judge) from .env; retries set here only
    tools/ middleware/ shared tools and middleware, referenced by name
    tracing.py         Langfuse (off without keys)
  workflows/
    ingestion/         handler.py (CLI), deps.py, agents.py, prompts/, actions/, sources/
  db/  storage/        tables and the blob store
```

## Setup (macOS)

Prerequisites: PostgreSQL 18 from the EDB installer, the Xcode Command Line Tools, [uv](https://docs.astral.sh/uv/), and a Gemini API key.

```bash
cp .env.example .env                 # then set GEMINI_API_KEY in .env
bash scripts/install_pgvector.sh     # builds pgvector; sudo asks for your Mac password
bash scripts/init_db.sh              # role + database + extension; asks for the postgres password
uv sync                              # Python 3.12 + dependencies
uv run python scripts/smoke_test.py  # checks Postgres, pgvector, embeddings and structured output
uv run alembic upgrade head          # creates the tables
```

`init_db.sh` generates the app's database password and writes it to `.env`. `.env` is gitignored.

## Ingestion

One command per pipeline step:

```bash
# 1. Record pages that were already fetched: verify each sha256, store the bytes, add artifacts rows
uv run ingest import-snapshot <snapshot folder>

# 2. Parse: artifacts → documents (clean text, doc type from the site category, content hash)
uv run ingest parse

# 3. Extract facts: documents → promotions (rules for fixed formats, Gemini structured output otherwise)
uv run ingest extract              # --only <url part> --force to redo one page

# 4. Chunk: documents → chunks of a corpus version (header line, one chunk per FAQ question,
#    jieba keyword text). Options: --segmenter none|jieba|bigram, --chunk-tokens, --no-header
uv run ingest chunk
```

Every step is safe to re-run. Parse skips pages that aren't HTTP 200, have no text (image-only), or duplicate another page, and records the reason in `documents.skip_reason`. Extract checks every output in code (dates must be quoted from the page, start ≤ end, numbers must appear on the page), retries once with the errors, and logs each run in `extractions`. Only outputs that pass become `promotions` rows.

A snapshot folder holds `sources.csv` and `raw/<slug>.html`. Raw pages are never committed; [`data/sources.csv`](data/sources.csv) lists each page's URL and sha256 instead.
