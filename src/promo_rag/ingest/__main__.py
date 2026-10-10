"""Ingestion commands, one per pipeline step.

uv run python -m promo_rag.ingest import-snapshot ~/Downloads/rag_poya_corpus
"""

import argparse
import sys
from pathlib import Path

from promo_rag.config import REPO_ROOT, get_settings
from promo_rag.db.session import new_session
from promo_rag.ingest.snapshot import load_snapshot, record_artifacts, write_manifest
from promo_rag.storage.blob import LocalBlobStore


def cmd_import_snapshot(args: argparse.Namespace) -> int:
    folder = Path(args.folder).expanduser()
    loaded = load_snapshot(folder)
    for reason in loaded.rejected:
        print(f"  rejected  {reason}")

    store = LocalBlobStore(get_settings().blob_dir)
    with new_session() as session:
        added = record_artifacts(session, store, loaded.pages)
    write_manifest(loaded.pages, REPO_ROOT / "data" / "sources.csv")

    print(
        f"{len(loaded.pages)} pages verified, {len(loaded.rejected)} rejected; "
        f"{added} new artifacts, {len(loaded.pages) - added} already recorded. "
        f"Wrote data/sources.csv."
    )
    return 1 if loaded.rejected else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m promo_rag.ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-snapshot", help="step 1: record pages that were already fetched")
    p.add_argument("folder", help="folder with sources.csv and raw/<slug>.html")
    p.set_defaults(func=cmd_import_snapshot)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
