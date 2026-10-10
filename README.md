# Retail Promotions RAG Assistant

A chat assistant that answers questions about a retailer's promotions, offers and products, in Chinese and English, from the retailer's public web pages. An LLM agent routes each question to SQL tools (count and list promotions by date, store or payment method) or to hybrid search (pgvector + full-text, rank fusion, reranking) for rules and fine print. Every answer cites its sources, and every design choice is measured on a golden question set.

*Work in progress.*

## Setup (macOS)

Prerequisites: PostgreSQL 18 from the EDB installer, the Xcode Command Line Tools, [uv](https://docs.astral.sh/uv/), and a Gemini API key.

```bash
cp .env.example .env                 # then set GEMINI_API_KEY in .env
bash scripts/install_pgvector.sh     # builds pgvector; sudo asks for your Mac password
bash scripts/init_db.sh              # role + database + extension; asks for the postgres password
uv sync                              # Python 3.12 + dependencies
uv run python scripts/smoke_test.py  # checks Postgres, pgvector, embeddings and structured output
```

`init_db.sh` generates the app's database password and writes it to `.env`. `.env` is gitignored.
