"""The ingestion workflow's entry point: one command per action (pipeline step).

uv run ingest import-snapshot ~/Downloads/rag_poya_corpus
uv run ingest parse
uv run ingest extract
"""

import argparse
import sys
from pathlib import Path

from promo_rag.agents.tracing import flush_traces
from promo_rag.config import REPO_ROOT
from promo_rag.workflows.ingestion import deps
from promo_rag.workflows.ingestion.actions.chunk import chunk_documents
from promo_rag.workflows.ingestion.actions.extract import extract_documents
from promo_rag.workflows.ingestion.actions.import_snapshot import (
    load_snapshot,
    record_artifacts,
    write_manifest,
)
from promo_rag.workflows.ingestion.actions.parse import parse_documents
from promo_rag.workflows.ingestion.index_config import IndexConfig


def cmd_import_snapshot(args: argparse.Namespace) -> int:
    folder = Path(args.folder).expanduser()
    loaded = load_snapshot(folder)
    for reason in loaded.rejected:
        print(f"  rejected  {reason}")

    with deps.session() as session:
        added = record_artifacts(session, deps.blob_store(), loaded.pages)
    write_manifest(loaded.pages, REPO_ROOT / "data" / "sources.csv")

    print(
        f"{len(loaded.pages)} pages verified, {len(loaded.rejected)} rejected; "
        f"{added} new artifacts, {len(loaded.pages) - added} already recorded. "
        f"Wrote data/sources.csv."
    )
    return 1 if loaded.rejected else 0


def cmd_parse(args: argparse.Namespace) -> int:
    with deps.session() as session:
        report = parse_documents(session, deps.blob_store(), rebuild=args.rebuild)

    print(
        f"{report.added} new documents, {report.new_versions} new versions, "
        f"{report.unchanged} unchanged."
    )
    print("By doc_type (status):")
    for key, n in sorted(report.by_doc_type.items()):
        print(f"  {n:4d}  {key}")
    if report.skipped:
        print("Skipped:")
        for url, reason in report.skipped:
            print(f"  {url}  ← {reason}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    with deps.session() as session:
        r = extract_documents(session, force=args.force, limit=args.limit, only=args.only)
    flush_traces()

    print(
        f"\nRun {r.run_id}. Extractions: {dict(r.by_status)} by method {dict(r.by_method)}; "
        f"{r.skipped_done} already done."
    )
    print(
        f"Promotions: {r.promotions_new} new, {r.promotions_merged} pages merged into an "
        f"existing promotion."
    )
    print(
        f"LLM tokens: {r.tokens_in:,} in, {r.tokens_out:,} out (≈ ${r.cost_usd:.4f} at paid prices)"
    )
    for url, status, errors in r.problems:
        print(f"  {status}: {url}")
        for attempt, errs in enumerate(errors, 1):
            for err in errs:
                print(f"      attempt {attempt}: {err}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    overrides = {
        k: v
        for k, v in {
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "segmenter": args.segmenter,
            "header": False if args.no_header else None,
        }.items()
        if v is not None
    }
    cfg = IndexConfig(**overrides)
    with deps.session() as session:
        r = chunk_documents(session, cfg, rebuild=args.rebuild)

    print(f"Corpus version {r.corpus_version_id} (fingerprint {r.fingerprint})")
    if r.already_built:
        print(f"Already built with these settings and inputs: {r.n_chunks} chunks. Nothing to do.")
        return 0
    counts = sorted(r.token_counts)
    print(
        f"{r.n_documents} documents → {r.n_chunks} chunks; {r.multi_chunk_docs} documents "
        f"have more than one.\nEstimated tokens per chunk: min {counts[0]}, median "
        f"{counts[len(counts) // 2]}, max {counts[-1]}."
    )
    print("Chunks by doc_type:", dict(sorted(r.by_doc_type.items())))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-snapshot", help="step 1: record pages that were already fetched")
    p.add_argument("folder", help="folder with sources.csv and raw/<slug>.html")
    p.set_defaults(func=cmd_import_snapshot)

    p = sub.add_parser("parse", help="step 2: artifacts → documents (clean text, doc type, hash)")
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="dev only: delete all documents (and their chunks/promotions) and parse again",
    )
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("extract", help="step 3: documents → checked facts → promotions")
    p.add_argument("--force", action="store_true", help="re-extract pages that are already done")
    p.add_argument("--limit", type=int, help="only the first N pages (for trying it out)")
    p.add_argument("--only", help="only pages whose URL contains this text, e.g. /news-detail/361/")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("chunk", help="step 4: documents → chunks of a corpus version")
    p.add_argument("--chunk-tokens", type=int, help="tokens per chunk (default 512)")
    p.add_argument(
        "--overlap-tokens", type=int, help="tokens repeated between chunks (default 100)"
    )
    p.add_argument("--segmenter", choices=["none", "jieba", "bigram"], help="default jieba")
    p.add_argument("--no-header", action="store_true", help="no 'title | dates | stores' line")
    p.add_argument("--rebuild", action="store_true", help="rebuild even if this version exists")
    p.set_defaults(func=cmd_chunk)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
