"""Step 1 (fetch) for pages that were already downloaded: verify, store, record.

A snapshot folder holds `sources.csv` (one row per fetched page, with its sha256) and
`raw/<slug>.html` (the bytes exactly as fetched). The live fetcher (M5) will write the
same `artifacts` rows, so every later step works the same for both.

Every fetch is recorded, including 404s: in a re-crawl, a 404 is how we learn a page
was removed. Parsing (step 2) decides which artifacts become searchable documents.
"""

import csv
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from promo_rag.db.models import Artifact
from promo_rag.storage.blob import BlobStore

# Columns from the crawler's list pages worth keeping for parsing.
FETCH_META_FIELDS = ("slug", "kind", "category", "list_date", "title")
# Columns published in the repo's data/sources.csv (the pages themselves are never committed).
MANIFEST_FIELDS = ("slug", "url", "sha256", "bytes", "fetched_at", "http_status")


@dataclass
class SnapshotPage:
    row: dict[str, str]
    data: bytes


@dataclass
class SnapshotLoad:
    pages: list[SnapshotPage] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # "slug: reason"


def load_snapshot(folder: Path) -> SnapshotLoad:
    """Read every page listed in sources.csv and check its bytes against the recorded sha256."""
    result = SnapshotLoad()
    with open(folder / "sources.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = folder / "raw" / f"{row['slug']}.html"
            if not path.exists():
                result.rejected.append(f"{row['slug']}: file missing")
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                result.rejected.append(f"{row['slug']}: bytes changed since the fetch (sha256)")
                continue
            result.pages.append(SnapshotPage(row=row, data=data))
    return result


def record_artifacts(session: Session, store: BlobStore, pages: list[SnapshotPage]) -> int:
    """Store the bytes and add one artifacts row per page. Returns how many rows were new.

    Safe to re-run: an existing (source_url, sha256) is left alone. Blobs are written
    before the commit; if the commit fails, the orphan blobs are harmless and get reused.
    """
    added = 0
    for page in pages:
        row = page.row
        key = store.put(page.data, ".html")
        stmt = (
            insert(Artifact)
            .values(
                sha256=row["sha256"],
                source_url=row["url"],
                final_url=row["final_url"] or None,
                fetched_at=datetime.fromisoformat(row["fetched_at"]),
                http_status=int(row["http_status"]),
                mime_type="text/html",
                byte_size=len(page.data),
                blob_key=key,
                fetch_meta={k: row[k] for k in FETCH_META_FIELDS if row.get(k)},
            )
            .on_conflict_do_nothing(index_elements=["source_url", "sha256"])
            .returning(Artifact.id)
        )
        if session.execute(stmt).scalar_one_or_none() is not None:
            added += 1
    session.commit()
    return added


def write_manifest(pages: list[SnapshotPage], path: Path) -> None:
    """Write the public list of sources (url + sha256), so others can re-fetch and verify."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for page in sorted(pages, key=lambda p: p.row["slug"]):
            writer.writerow({k: page.row[k] for k in MANIFEST_FIELDS})
