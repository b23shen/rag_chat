"""Try a search from the terminal.

uv run search "週五用街口支付有什麼優惠?"
uv run search "Can I bring my dog?" --mode vector -k 3
uv run search "退貨" --mode keyword --version 02fc002a      # another corpus version, by id prefix
uv run search "可以帶狗嗎?" --mode rerank                     # needs COHERE_API_KEY in .env
"""

import argparse
import sys
import time

from promo_rag.db.session import new_session
from promo_rag.search.rerank import RerankError
from promo_rag.search.retrieve import SearchError, find_version, search

SITE = "https://www.poya.com.tw"


def main() -> int:
    parser = argparse.ArgumentParser(prog="search")
    parser.add_argument("query")
    parser.add_argument(
        "--mode", choices=["vector", "keyword", "hybrid", "rerank"], default="hybrid"
    )
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--version", help="corpus version id or its first characters")
    args = parser.parse_args()

    with new_session() as session:
        try:
            version_id = find_version(session, args.version) if args.version else None
            start = time.perf_counter()
            hits = search(session, args.query, mode=args.mode, k=args.k, version_id=version_id)
            ms = (time.perf_counter() - start) * 1000
        except (SearchError, RerankError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        session.commit()  # keep the question's embedding in the cache

    for h in hits:
        ranks = ""
        if args.mode in ("hybrid", "rerank"):
            ranks = f"vector #{h.vector_rank or '-'}, keyword #{h.keyword_rank or '-'}"
            if args.mode == "rerank":
                ranks = f"hybrid #{h.hybrid_rank}: {ranks}"
            ranks = f"  ({ranks})"
        body = h.text.split("\n", 1)[-1].replace("\n", " ")
        print(f"{h.rank}. {h.score:.4f}  {h.source_url.removeprefix(SITE)}  {h.title}{ranks}")
        print(f"   {body[:90]}")
    if not hits:
        print("No results.")
    print(f"{args.mode}, {ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
