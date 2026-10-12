# M1 handoff: ingestion as a workflow

*2026-10-11 · Status: done · 116 tests passing*

## What M1 delivers

A workflow that takes a **dataset of fetched retailer pages** and produces an **active, searchable corpus version**: clean documents, checked promotion facts in SQL, and chunks with jieba keyword text and `gemini-embedding-2` vectors in pgvector. It runs over HTTP (`POST /workflows/ingestion/start`, then `/status` and `/content`), is executed by a worker from a Postgres-backed queue, and every step is safe to re-run.

The corpus is 133 public pages from POYA 寶雅, a Taiwanese retailer (Traditional Chinese).

## Results

| Step | Output |
|---|---|
| Import | 133 pages, each verified against its recorded sha256 (0 rejected) |
| Parse | 118 active documents; 15 skipped, each with a recorded reason (1 HTTP 404, 10 image-only, 3 exact duplicates, 1 page that repeats three others) |
| Extract | 101 / 101 extractions pass the checks → 100 promotions (one campaign appears on two pages and is merged). 65 store notices read by rules (0 LLM calls), 36 campaigns by `gemini-3.1-flash-lite`. 94 have dates; 6 pages state no period and honestly have none. |
| Chunk | 232 chunks; median ~360 estimated tokens, largest 467 real tokens (limit 512); 39 pages have more than one |
| Embed | 232 / 232 embedded in 5 API calls; every vector has norm 1.0. A rebuild with unchanged text: 0 API calls, 0.7 s |
| Activate | Corpus version `8348a0f5` active (rebuilt by the edit-and-undo test below); GIN and HNSW indexes valid |
| Workflow | Runs over HTTP; a re-run of the full dataset takes ~4 s (nothing to redo). Editing one page and re-ingesting it: 1 LLM call, 231 / 232 embeddings from the cache |

**Cost:** under US$0.10 at paid prices for all of M1's development runs ($0 on the free tier). A full extraction is ~40k input / 8k output tokens (≈ $0.02); embedding the corpus is ~71k tokens (≈ $0.014).

**Sanity check (vector search only, before M2):** "週五用街口支付有什麼優惠?" → the JKOPay Friday offer (0.79); "可以帶寵物進店嗎?" → the pets FAQ (0.83); the English question "Which new stores open in October?" → exactly the three October store openings.

## How it works

```
POST /datasets, /datasets/{id}/artifacts      (or: uv run ingest import-snapshot <folder>)
        │
POST /workflows/ingestion/start  ──►  platform.workflow_runs (queued)
        │                                       │  worker: SELECT … FOR UPDATE SKIP LOCKED
        ▼                                       ▼
   runbook.yaml:  parse ─► extract ─► chunk ─► embed ─► activate      (each recorded in action_runs)
```

| Action | What it does | Key rules |
|---|---|---|
| **parse** | The dataset's pages → `documents` | Site adapter with 6 page templates, selectors scoped to `<main>`; strips comments, scripts, menus and hidden elements (prompt-injection defense); NFKC normalization; doc type from the site's own category; `content_hash` finds duplicates; new content for a URL = new version, old one archived |
| **extract** | Promotion and store-notice pages → `promotions` | Rules first for fixed formats; otherwise the `fact_extractor` agent (Gemini structured output → Pydantic). Checks in code: dates must appear inside the quoted period text, start ≤ end, within a year of the publish date, numbers must appear on the page. One retry with the errors; failures go to `extractions` as `needs_review`, never into `promotions`. Same title + dates on two pages = one promotion with two sources. |
| **chunk** | Active documents → chunks of a corpus version | Whole sentences packed to 512 tokens with 100 overlap; one chunk per FAQ question; every chunk starts with a header (`title \| period or 公告 <date> \| stores \| section`); keyword text split by jieba + a POYA word list |
| **embed** | Fill the version's embeddings | `title: … \| text: …` format; one `Content` per text; cache keyed by (model, dims, sha256 of the input); checks one vector per text and the dims; resumable |
| **activate** | Make the version the one search reads | Refuses unless every chunk is embedded with the right model and dims, every page has chunks and nothing changed since the build; one transaction; at most one active version (database constraint); then `ANALYZE` and query-plan checks |

**Data model:** one Postgres database, two schemas.
- `platform`: `datasets`, `dataset_artifacts`, `artifacts`, `workflows`, `workflow_runs`, `action_runs`
- `rag`: `documents`, `promotions`, `promotion_sources`, `extractions`, `corpus_versions`, `chunks`, `embedding_cache`

Raw page bytes live in a blob store (a local folder now, GCS later), keyed by sha256. All migrations (0001–0007) are plain SQL in [`migrations/sql/`](../../migrations/sql/).

## Design decisions, and why

- **Two stores per page.** Facts with types (dates, stores, minimum spend) go into `promotions` for SQL; the text goes into chunks for search. "How many promotions in October?" needs a count, which top-k search can't give.
- **Rules before the LLM, and code checks after it.** The model's answer is only trusted after checks it can't talk its way past. A failed check is logged, not written.
- **Content hashes everywhere.** Raw bytes (`artifacts.sha256`), clean text (`content_hash`), index settings (fingerprint), prompt + schema (prompt version) and embedding inputs (cache key): "is this exactly the same as before?" is always one comparison.
- **Corpus versions.** A version is its settings (fingerprint) plus its inputs (each page's text and header facts). Building is deterministic, so the same settings and inputs are never rebuilt, and versions sit side by side for A/B tests (a bigram version is already built for M2).
- **Headers on chunks.** A chunk cut from the middle of a page still says which promotion and period it belongs to.
- **Retries in one place.** The model client allows at most 2 retries (LangChain's default is 6); a fallback model handles what's left. Stacked retry layers multiply the worst-case wait.
- **The run row is the queue.** No Redis or Celery; a crash can't silently lose a run, and `SKIP LOCKED` lets several workers share the queue.
- **One way to call an agent.** `invoke_agent(name, messages, variables=, response_schema=, trace=)`; prompts are Markdown with `{{slots}}`; each workflow defines its agents (model role, prompt, schema); tools and middleware are shared.

## Lessons from building it

- **`gemini-embedding-2` treats a plain list of strings as one input** and returns one blended vector, with no error. Found in the smoke test; the embed step now checks the vector count.
- **A wrong check is worse than no check.** A check that searched only the page body rejected a correct minimum spend that was in the title, and on the retry the model *removed* the correct value to pass. Checks now search everything the model saw.
- **A real quote isn't enough evidence.** The model quoted "活動期間每週五~週日" (every Fri–Sun) and took a voucher deadline as the end date. Dates must now appear inside the quoted period.
- **Fields without checks drift.** Two runs at temperature 0 agreed on 100 / 101 dates (checked) but differed on 8 pages' unchecked fields (`promo_type`, `stores`). Exclusion lists ("POYA Pay not accepted at…") were read as "only at these stores"; there's now a separate `excluded_stores` field.
- **Hashes can't find near-duplicates.** One campaign on two differently worded pages, and a FAQ page that repeats three others, needed rules of their own.
- **Measure before adopting.** jieba's larger dictionary looked like the fix for Traditional Chinese but broke more brand words than it fixed; a 36-word list fixed the words that matter.
- **Versioning needs an "edit, then undo" test.** Uploading an edited page and then the original again found three bugs, all fixed:
  1. *A → B → A ended on B.* The original bytes matched the existing artifact, which still had its old fetch time. Artifacts now have `last_seen_at`, moved forward on every fetch.
  2. *Old facts survived a page update.* The archived version's promotion stayed linked, and the edited page merged into it. Archiving a document now unlinks its promotions.
  3. *A corpus version matched by content, not by row.* The undo had the original text, so chunk reused the original version, whose chunks point at the archived v1 row. **The activate gate refused** ("1 active page has no chunks"), so nothing bad went live. The inputs hash now includes document ids.
- **HNSW isn't used at this size,** and shouldn't be: filtering 232 rows by version and sorting exactly is cheaper and exact. At scale, a version filter on HNSW needs pgvector 0.8's iterative scan, a per-version index, or archiving retired versions.

## Known gaps

- **No live crawler yet.** Pages come from a snapshot or an upload. The polite fetcher (robots.txt, rate limit) and a daily re-crawl come in M5.
- **Image-only content isn't read:** 8 catalogs, one card-benefits page, and the offers on store-opening posters. A vision step is a later, measured experiment.
- **6 promotions have no dates** because their pages state none. The M3 SQL tools must decide how to treat them.
- **Extraction accuracy isn't measured yet;** that needs hand-labeled pages. `promo_type` is the least consistent field.
- **The store-list page only has 9 stores** in its HTML (the rest load with JavaScript); store questions rely on the 65 store notices.
- **The worker has no heartbeat.** If it dies mid-run, the run stays `running`. Next: a heartbeat column and re-queueing stale runs.
- **The API has no authentication** (local only). M5 adds IAM or an API key before deploying.
- **Langfuse tracing is wired but off** until keys are set in `.env`.
- **No database integration tests.** Unit tests cover the logic; the workflow was verified end to end by hand (see below).

## Verify it yourself

```bash
uv run pytest -q                                   # 116 tests
uv run alembic check                               # models match the database
uv run serve &  uv run worker --once               # with a queued run, or:
uv run ingest import-snapshot <folder>             # → dataset id
curl -X POST localhost:8000/workflows/ingestion/start -H 'content-type: application/json' -d '{"dataset_id": "<id>"}'
uv run worker --once && curl localhost:8000/runs/<run_id>/content
```

Checked by hand on 2026-10-11:
- bad requests (unknown dataset, misspelled param) → 400
- content before a run finishes → 409
- a full re-run → every action reports nothing to redo
- `segmenter=bigram` → new version, 232 / 232 embeddings from the cache, activated; `segmenter=jieba` → switched back
- uploading a page that's already stored → `new: false`, and a run on that one-page dataset parses only that page
- **edit and undo:** an edited copy of one page → new document version, old promotion removed, 1 LLM call, new corpus version with 231 / 232 embeddings from the cache, activated; the original page again → a third document version with the original text and a new corpus version built entirely from the cache (0 API calls)

## Next: M2, search and eval

A golden question set (counts, lists, dates, rules, unanswerable, multi-turn; Chinese and English), the search tool in four modes (vector, keyword, hybrid with rank fusion, hybrid + reranker), and the first results table (Hit@5, MRR, nDCG@5, p95 latency). The jieba and bigram corpus versions are both built, so the segmenter comparison is ready to run.
