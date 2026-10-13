# M2 handoff: search and eval

*2026-10-12 · Status: done · 164 tests passing*

## What M2 delivers

A **golden question set**, a **search function in four modes**, and an **eval runner** that measures every mode and setting on the same questions, question by question. The default search, `rerank`, takes vector search's top 20 chunks and re-orders them with Cohere Rerank. Keyword search and hybrid search are built and measured, but on this corpus they made results worse, so they stay as modes for experiments only.

```bash
uv run search "週五用街口支付有什麼優惠?"     # the default: vector top 20 → Cohere rerank → top 5
uv run eval-search                            # every mode on the 102 golden questions
```

## Results

All numbers are on the active corpus version `5da896fe` (jieba, 232 chunks) and the 82 answerable golden questions. A hit is a chunk that answers the question. Latency is search time with the question's embedding already cached.

| Mode | Hit@5 | MRR@10 | nDCG@5 | MRR@10, 15 exact-match questions | p50 / p95 ms |
|---|---|---|---|---|---|
| **rerank (default)** | 0.99 | **0.94** | **0.93** | **1.00** | 391 / 545 |
| vector | 1.00 | 0.91 | 0.91 | 0.91 | 7 / 10 |
| hybrid, best setting (bigram, vector weight 5) | 0.96 | 0.84 | 0.82 | 0.92 | 9 / 13 |
| hybrid, equal weights (jieba) | 0.95 | 0.74 | 0.74 | 0.97 | 11 / 14 |
| keyword (bigram) | 0.61 | 0.45 | 0.47 | 0.87 | 2 / 4 |

**Rerank vs vector:** +0.03 MRR, but the 95% range (−0.03 to +0.09) includes 0, so the two can't be told apart on 82 questions. Rerank is the default because of two things vector can't do. It is best on exact names and numbers (1.00), and its scores can flag "nothing relevant" (below).

**Unanswerable questions (20):** a cutoff on the top result's score that loses no answerable question catches 7 of 20 with rerank (cutoff 0.61), and 1 of 20 with vector. Near misses ("全聯 points", "Starbucks stars", "玉山 card") score 0.89–0.94, as high as real answers, so only the answering LLM can decline those.

**Experiments, each compared question by question with vector:**

| Experiment | Result | Decision |
|---|---|---|
| Hybrid vs vector | Hybrid lower at every setting; keyword helped 1 question and hurt 6 to 13 (44-question run) | Keyword and hybrid: experiments only |
| Segmenter for keyword | Bigram beats jieba: keyword Hit@5 0.52 → 0.61 | Use bigram if keyword returns |
| Common-word filter (10%, 20%, 30%, off) | 20–30% best, off worst; small gaps | Keep 20% |
| Vector weight in RRF (1 → 5) | Hybrid MRR 0.77 → 0.84, still below vector | n/a |
| Rerank candidates: vector vs hybrid | Same MRR (0.93, 44-question run) | Vector: simpler, no keyword |
| Question prefix: `search result` vs `question answering` vs none | +0.01 and −0.00, ranges include 0; `search result` best on exact-match questions | Keep `search result` |

Every run is saved in [`evals/results/`](../../evals/results/) (Markdown tables + JSON with every question's results and settings). **Cost:** about 250 Cohere trial calls (of 1,000 a month), and a few Gemini embedding calls for the questions (chunk embeddings all came from the cache): under US$0.01.

## How it works

```
question
   │
vector search      → top 20 candidates (pgvector, cosine, exact scan of the version's chunks)
   ▼
reranker           → Cohere rerank-v4.0-pro reads the question and each chunk together
   ▼
top k              → the LLM that writes the answer (M3)
```

| Module | What it does | Key rules |
|---|---|---|
| [`search/retrieve.py`](../../src/promo_rag/search/retrieve.py) | `search(query, mode=rerank \| vector \| keyword \| hybrid)` → hits (chunk, page, score, ranks) | The question is embedded and split with **the settings its corpus version was built with**, and refuses a jieba version built with another word list. Knobs live in `SearchSettings` |
| [`search/rerank.py`](../../src/promo_rag/search/rerank.py) | Cohere Rerank v2 over httpx | At most 2 retries (rate limits, server errors); scores put back in input order and checked (one per document). If it fails, search keeps vector's order and marks `reranked=False`; the eval passes `fallback=False` |
| keyword mode | Postgres full-text on the segmented text | Any question word may match (OR). Words in more than 20% of chunks are dropped, because `ts_rank` has no IDF |
| hybrid mode | Reciprocal rank fusion of both lists' top 20 | `weight / (60 + rank)`: ranks, not scores, since the score scales differ |
| [`evals/golden.yaml`](../../evals/golden.yaml) | 102 questions: rule 26, exact 15, date 12, list 11, count 8, multi-turn 10, unanswerable 20; 40 in English | Each label is a **page path plus a short quote** (164 labels, 89 pages). A chunk is relevant if it comes from the page and contains the quote |
| [`evals/search_eval.py`](../../src/promo_rag/evals/search_eval.py) | Runs **variants** (mode × corpus version × setting) on the same questions | Compares each with the first variant, with a bootstrap 95% range; reports no-answer cutoffs; one untimed run per variant first |

## Design decisions, and why

- **Labels are pages plus quotes, not chunk ids.** Chunk ids change with every corpus version, and chunk numbers move when the chunk size changes. A quote is found in whichever chunk holds it, so one label set grades every version. A test checks that every quote sits inside one chunk of the active version, which catches typos and quotes cut by a chunk boundary.
- **Grading is per chunk.** 55 of the 82 answerable questions point at pages with several chunks (one FAQ page has 15). Page-level grading would count a wrong chunk from the right page as a hit.
- **Compare question by question, with a range.** With 44 questions, a gap of 0.07 in MRR was noise. Each variant's difference from the baseline is bootstrapped (2,000 resamples of the questions), and a range that includes 0 is reported as "no clear difference".
- **Variants in one run.** Every mode and setting sees the same questions in the same process, so the comparisons are paired and a whole experiment is one command.
- **Latency is search time only.** Question embeddings are fetched in one batch first, each variant runs once untimed (loading jieba and word counts), and the reranker's rate limit is waited out before the timer starts. The embedding call for a new question adds one API call on top in real use.
- **Rerank is the default, fed by vector.** It ranks at least as well as vector, is best on exact matches, and gives a score that means the same thing across questions. Feeding it hybrid's list instead gave the same scores with more moving parts.
- **Search falls back; the eval doesn't.** A trial key allows 10 calls a minute, and chat may search several times a turn, so search must not stop when Cohere does. In the eval, a fallback would pass vector's order off as rerank's, so it raises.
- **Follow-ups are searched with the earlier user turns** joined to the question. This is a simple stand-in; the M3 agent should rewrite follow-ups into standalone questions.

## Lessons from building it

- **Hit@5 saturates on a small corpus.** With 232 chunks, vector finds an answer in the top 5 for every question, so Hit@5 can't separate methods. MRR, nDCG and the paired range do.
- **Equal-weight RRF is only as good as its weaker list.** Rank 1 in a noisy keyword list counts as much as rank 1 in a good vector list; chunks found by both lists at middle ranks then beat vector's correct first result. On rule-01, vector put the answer first and hybrid put it eighth.
- **A test set can't show what it doesn't ask.** The first 50 questions had no exact-match questions, so they could show keyword search's cost but not its value. Fifteen exact questions showed its value (hybrid 0.97 vs vector 0.91 there) and that rerank gives the same benefit without it (1.00).
- **Postgres `ts_rank` has no IDF.** Without the common-word filter, chunks full of 2026, 活動 and 的 outrank the chunk that holds the question's rare word.
- **Vector similarity isn't a confidence.** Unanswerable questions score 0.68–0.80 and answerable ones from 0.68 up, so no vector cutoff works. Rerank's scores separate a third of them.
- **The reranker judges the topic, not conditions.** For "clearance sales that started in May", it ranked eight March and April clearance notices above the May ones. Dates and counts are work for SQL tools, not for ranking.
- **Building two test versions in a row lost the first.** The chunk step removes drafts without embeddings, so a test version must be embedded right after it is built (now in the README).
- **Raw SQL returns pgvector values as text**, not lists; parse them (`embedding::text` + JSON) when reading vectors outside the ORM.

## Known gaps

- **The golden set was written from the pages** by one author (reviewed by a second). Its questions are still easier and closer to the pages' wording than real users' questions, and 102 questions only separate gaps larger than about 0.05.
- **The no-answer cutoff is a hint, not a rule.** It was fitted on the same 20 questions it is scored on, and search doesn't apply it; the M3 agent should combine it with the LLM's own judgment.
- **Count and list questions are graded leniently.** A hit needs one answering chunk, but "how many stores…" needs all of them. Recall of every relevant page isn't measured; those questions belong to the M3 SQL tools.
- **Rerank needs a production key before deploying.** The trial key allows 10 calls a minute and 1,000 a month and may not be used in production. Rerank also adds about 390 ms (545 ms p95) per search.
- **Learned sparse search (BGE-M3) wasn't tried.** It needs PyTorch and a 2.2 GB model; keyword and hybrid modes are kept so it can be measured later.
- **The eval runs from the command line only**, not as a workflow over HTTP like ingestion.
- **Data quirks found on the site:** events pages 47 and 48 carry each other's brand lists; winner notices 364 and 365 name each other's foundation; the GIFT page's December qualifying range ends 2026/12/14, probably a typo for 2026/11/14. The golden set labels follow the pages as published.

## Verify it yourself

```bash
uv run pytest -q                                     # 164 tests (the database tests skip without Postgres)
uv run search "Can I bring my dog?" -k 3             # rerank; shows each chunk's rank before reranking
COHERE_API_KEY= uv run search "可以帶狗嗎?" -k 2      # the fallback: vector order, and a note saying so
uv run eval-search --modes vector hybrid             # fast (no reranker); hybrid shows ΔMRR around −0.18
uv run eval-search --modes vector rerank             # about 10 minutes on a trial key
```

Checked on 2026-10-12: every golden quote is inside one chunk of the active version; the fallback works with an empty key; the default index settings still have the active version's fingerprint (`515f970c43e9f17a`).

## Next: M3, the agent

The chat agent that answers from the corpus: SQL tools for counts, lists and dates (`count_promotions`, `list_promotions`, `get_promotion`, `resolve_dates`) next to `search_promotion_text` (this M2 search), rewriting follow-up questions before searching, answers with citations, and declining when the sources don't answer (the rerank score as a first signal). The golden set's `answer` fields are ready for an answer eval.
