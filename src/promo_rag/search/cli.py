"""Try a search from the terminal.

uv run search "週五用街口支付有什麼優惠?"
uv run search "Can I bring my dog?" --mode vector -k 3
uv run search "退貨" --mode keyword --version 02fc002a      # another corpus version, by id prefix
"""

import argparse
import sys
import time
import uuid

from sqlalchemy import text

from promo_rag.db.session import new_session
from promo_rag.search.retrieve import SearchError, search

SITE = "https://www.poya.com.tw"


def main() -> int:
    parser = argparse.ArgumentParser(prog="search")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--version", help="corpus version id or its first characters")
    args = parser.parse_args()

    with new_session() as session:
        try:
            version_id = _version_id(session, args.version) if args.version else None
            start = time.perf_counter()
            hits = search(session, args.query, mode=args.mode, k=args.k, version_id=version_id)
            ms = (time.perf_counter() - start) * 1000
        except SearchError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        session.commit()  # keep the question's embedding in the cache

    for h in hits:
        ranks = ""
        if args.mode == "hybrid":
            ranks = f"  (vector #{h.vector_rank or '-'}, keyword #{h.keyword_rank or '-'})"
        body = h.text.split("\n", 1)[-1].replace("\n", " ")
        print(f"{h.rank}. {h.score:.4f}  {h.source_url.removeprefix(SITE)}  {h.title}{ranks}")
        print(f"   {body[:90]}")
    if not hits:
        print("No results.")
    print(f"{args.mode}, {ms:.0f} ms")
    return 0


def _version_id(session, prefix: str) -> uuid.UUID:
    ids = session.scalars(
        text("SELECT id FROM corpus_versions WHERE id::text LIKE :p"), {"p": prefix + "%"}
    ).all()
    if len(ids) != 1:
        raise SearchError(f"{len(ids)} corpus versions start with {prefix!r}")
    return ids[0]


if __name__ == "__main__":
    sys.exit(main())
