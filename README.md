# Retail Promotions RAG Assistant

A chat assistant that answers questions about a retailer's promotions, offers and products, in Chinese and English, from the retailer's public web pages. An LLM agent routes each question to SQL tools (count and list promotions by date, store or payment method) or to hybrid search (pgvector + full-text, rank fusion, reranking) for rules and fine print. Every answer cites its sources, and every design choice is measured on a golden question set.

*Work in progress.* Done: **M1, ingestion as a workflow** ([handoff](docs/handoffs/M1_ingestion.md)) and **M2, search and eval** ([handoff](docs/handoffs/M2_search_eval.md)). Next: M3, the agent.

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
  search/              word splitting (jieba), embeddings + cache, and search itself
    retrieve.py        search(query, mode=rerank (default) | vector | keyword | hybrid)
    rerank.py          Cohere Rerank client
    cli.py             `uv run search "…"`
  evals/               the golden question set's loader (the questions are in evals/golden.yaml)
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

`init_db.sh` generates the app's database password and writes it to `.env`. `.env` is gitignored. The `rerank` search mode also needs a Cohere key (`COHERE_API_KEY`; a free trial key works).

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

**Updating a page vs adding one.** A page is identified by its `source_url`, and a corpus version's `n_documents` counts pages (the active, latest document per URL), not rows:

| You upload | Result | `n_documents` |
|---|---|---|
| New content for a URL already in the corpus | A new document version; the old one is archived and its promotion removed | unchanged |
| A new URL whose text matches an existing page | Recorded, but skipped as `duplicate of <url>` | unchanged |
| A new URL with new content | A new page: document, promotion (if any) and chunks | +1 |
| Bytes already stored for that URL | Same artifact (`"new": false`), its `last_seen_at` moves forward. If they're an older version (A → B → A), that text becomes current again | unchanged |

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

## Facts for the SQL tools

The M3 SQL tools count and list promotions from the `promotions` table. The extract step fills that table, so its facts decide which questions SQL can answer.

**Each page type has its own path.** The site's category decides the page type ([`sources/poya.py`](src/promo_rag/workflows/ingestion/sources/poya.py)). Each type with facts has its own extractor ([`facts.EXTRACTORS`](src/promo_rag/workflows/ingestion/facts.py)): campaign pages go to the LLM with `PromotionFacts`, and store notices to rules (or the LLM, if the rules don't match). Every other type, including a category we haven't seen (`other`), is search-only: search finds its pages, but SQL can't count or list them. Add an extractor for a type only when the eval shows SQL questions about it failing.

**Tried and dropped: one general schema by default.** Hard-coded types looked fragile, so we tried a mix: one general LLM schema for every type without its own extractor, campaign pages included. It added a "does this page have a promotion" flag and two common columns, `weekdays` (每週五) and `payment_kinds` (credit card, mobile payment, …).

- All 101 pages passed the checks, but the campaign pages' answers got worse. A field changed on 18 of the 36 pages. Each page gave the same answer 3 times out of 3 with each schema, so the schema caused it, not chance.
- The old answers were better on most of them. 11 pages lost `app` from `channels` (page 331, for one, has "POYA寶雅APP操作說明"). The 台新 offer lost `POYA支付`, though the page requires it. LINE POINTS and 小樹點 rewards became `cashback` instead of `points`.
- The new columns weren't needed either. Only 4 offers are limited to some weekdays and 7 to some payment methods, and their titles say 週五 or 信用卡, so search finds them.

So the per-type schemas stay.

**Kept: `cities`.** "New stores in Kaohsiung in 2026" needs all six stores, and search returns only the top 5. Store notices fill `cities` from the store's address, by rules (no LLM): all 65 have one. Campaign pages leave it empty.

**Lesson:** adding fields to a schema can change the LLM's answers on fields you didn't touch. Before keeping a schema change, compare the extractions page by page, and run each page more than once.

## Search

Search reads the active corpus version's chunks and returns the top chunks for a question. The default mode, `rerank`, is a funnel: a cheap search finds candidates, then a slower, more accurate model puts them in order.

```
question
   │
vector search       → top 20 candidates: cheap, finds the right chunk almost always
   ▼
reranker            → reads the question and each chunk together, re-orders them
   ▼
top 5               → the LLM that writes the answer
```

If the reranker fails (no key, a trial key's limit of 10 calls a minute, an outage), search keeps vector's order and says so (`Hit.reranked = False`), so it never stops working.

| Mode | How it ranks | Score |
|---|---|---|
| `rerank` (default) | Vector's top 20, re-ordered by Cohere Rerank (`rerank-v4.0-pro`) | relevance from 0 to 1 |
| `vector` | Closeness of the question's embedding to each chunk's (pgvector, cosine) | cosine similarity, higher is closer |
| `keyword` | Postgres full-text search on the split text; any question word may match. Words found in more than 20% of chunks (的, 活動, 2026 …) are ignored, because Postgres ranking doesn't weigh rare words higher | `ts_rank`; compare only within one search |
| `hybrid` | Vector's and keyword's top 20, merged by reciprocal rank fusion: each chunk scores `weight / (60 + rank)` per list it appears in | at most 2/61 ≈ 0.0328 (first in both) |

**Why not hybrid?** On the golden set (82 answerable questions), keyword search made results worse: hybrid's MRR was 0.74–0.84 against vector's 0.91, whatever the segmenter (jieba or bigram), word filter (10%–off) or vector weight (1–5). Keyword does help where it should, on the 15 exact-match questions (phone numbers, codes, brand names): hybrid 0.97 vs vector 0.91. But rerank reaches 1.00 on those without it. So keyword and hybrid stay as modes for experiments (e.g. a learned sparse model such as BGE-M3), not for answers.

**Why rerank?** It ranks a little better than vector (MRR 0.94 vs 0.91; the 95% range, −0.03 to +0.09, still includes 0) and is the best on exact-match questions. Its scores also mean the same thing across questions, so a low top score can flag "nothing relevant": a cutoff of 0.61 catches 7 of the 20 unanswerable questions without losing an answerable one, where vector's scores catch 1. Near misses ("全聯 points", "Starbucks stars", "玉山 card") score as high as real answers (0.89–0.94), so the answering LLM must catch those.

```bash
uv run search "週五用街口支付有什麼優惠?"                 # rerank
uv run search "Can I bring my dog?" --mode vector -k 3
uv run search "退貨" --mode keyword --version <id>        # another corpus version, by id prefix
```

Each line of output is one chunk (a page can appear more than once): rank, score, page, title, and the start of the chunk's text. Rerank also shows each chunk's rank before reranking, and hybrid its rank in the vector and keyword lists.

The question is split into words and embedded with the settings its corpus version was built with, so a question and the chunks it is compared with are always processed the same way.

## Search eval

[`evals/golden.yaml`](evals/golden.yaml) holds 102 questions: rules, exact names and numbers, dates, lists, counts, multi-turn follow-ups, and 20 unanswerable ones (many of them near misses); 40 are in English. Each one names the pages that answer it and a short quote from each page. A search result chunk counts as relevant if it comes from one of those pages and contains the quote, so the labels work on every corpus version, whatever its chunk ids or boundaries.

```bash
uv run eval-search                          # all modes, the active corpus version (rerank takes ~5 min on a trial key)
uv run eval-search --modes vector hybrid    # some modes (the first is the baseline)
uv run eval-search --version <active id> <test id> --max-word-share 0.1 0.2 1.0   # compare variants
```

To compare an index setting (segmenter, chunk size, question format), build a test version and embed it right away: building the next one removes any draft that has no embeddings yet. Unchanged chunk text comes from the embedding cache, so this costs no API calls. The test version is never activated, so search doesn't read it.

```bash
uv run ingest chunk --segmenter bigram          # or --query-format qa, --chunk-tokens 384, …; prints the id
uv run ingest embed --version <test id>
```

Flags that take several values make one variant per combination (mode × corpus version × setting). For each variant it reports Hit@5 and Hit@20 (a chunk that answers the question is in the top 5 / 20), MRR@10 (1 / the rank of the first one), nDCG@5 (how close the top 5 is to the best possible order) and search latency.

Every variant is compared with the first one question by question, with a 95% range for the difference: draw the answerable questions at random, with repeats, 2,000 times, and keep the middle 95% of the results. A range that includes 0 means the two can't be told apart on this many questions; with 82 answerable questions, one question moves Hit@5 by 0.012. The unanswerable questions are searched too, to find the top-score cutoff (vector and rerank only) that best flags "nothing relevant". Results are saved to `evals/results/` as Markdown tables and JSON (every question's results and each variant's settings).
