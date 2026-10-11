"""Step 2 (parse): artifacts → documents.

For the latest fetch of each URL:
  1. Parse the HTML into clean text with the site adapter.
  2. Decide the doc_type from the site's own category (no LLM needed).
  3. Hash the text (content_hash) and decide the status:
       skipped   not HTTP 200, on the skip list, no text (image-only), or a duplicate
       active    everything else
  4. Write a documents row. If the URL already has a document:
       same content_hash and status  → nothing to do
       different                     → new version; the old one becomes archived

Skipped pages still get a row with the reason, so "why isn't X searchable?" has an answer.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from promo_rag.db.models import Artifact, Chunk, Document, Promotion, PromotionSource
from promo_rag.storage.blob import BlobStore
from promo_rag.workflows.ingestion.html_text import content_hash, detect_language
from promo_rag.workflows.ingestion.sources import poya

# Less body text than this means the page's content is in images (catalogs, banners).
MIN_TEXT_CHARS = 20


@dataclass
class ParseReport:
    added: int = 0
    new_versions: int = 0
    unchanged: int = 0
    by_status: Counter = field(default_factory=Counter)
    by_doc_type: Counter = field(default_factory=Counter)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (url, reason)


def parse_documents(session: Session, store: BlobStore, rebuild: bool = False) -> ParseReport:
    if rebuild:
        _delete_documents(session)

    report = ParseReport()
    # Latest version per URL that is already active, for duplicate detection: hash → url.
    first_url_for_hash: dict[str, str] = dict(
        session.execute(
            select(Document.content_hash, Document.source_url).where(
                Document.is_latest, Document.status == "active"
            )
        ).all()
    )

    for artifact in _latest_artifacts(session):
        url = artifact.source_url
        page = poya.parse_page(store.get(artifact.blob_key), url)
        meta = artifact.fetch_meta or {}
        category = page.category or meta.get("category")
        text_hash = content_hash(page.text)
        status, reason = _decide_status(artifact, page, text_hash, first_url_for_hash)
        if status == "active":
            first_url_for_hash.setdefault(text_hash, url)

        doc_type = poya.doc_type_for(category, url)
        report.by_status[status] += 1
        report.by_doc_type[f"{doc_type} ({status})"] += 1
        if reason:
            report.skipped.append((url, reason))

        current = session.scalars(
            select(Document).where(Document.source_url == url, Document.is_latest)
        ).one_or_none()
        if current and current.content_hash == text_hash and current.status == status:
            report.unchanged += 1
            continue
        if current:
            # The partial unique index allows one latest row per URL, so retire this one first.
            current.is_latest = False
            current.status = "archived"
            session.flush()
            report.new_versions += 1
        else:
            report.added += 1

        session.add(
            Document(
                artifact_id=artifact.id,
                source_url=url,
                title=page.title,
                language=detect_language(page.title + page.text),
                doc_type=doc_type,
                category=category,
                published_date=page.published_date or _list_date(meta),
                content_hash=text_hash,
                full_text=page.text,
                image_urls=page.image_urls,
                version=current.version + 1 if current else 1,
                previous_version_id=current.id if current else None,
                status=status,
                skip_reason=reason,
            )
        )
    session.commit()
    return report


def _latest_artifacts(session: Session) -> list[Artifact]:
    """The newest fetch of each URL, shortest URL first.

    Shortest-first makes duplicate handling deterministic: /member/ is kept and
    /member-detail/5/ is marked as its duplicate.
    """
    latest = (
        select(Artifact)
        .distinct(Artifact.source_url)
        .order_by(Artifact.source_url, Artifact.fetched_at.desc())
        .subquery()
    )
    rows = session.scalars(select(Artifact).join(latest, Artifact.id == latest.c.id)).all()
    return sorted(rows, key=lambda a: (len(a.source_url), a.source_url))


def _decide_status(
    artifact: Artifact, page: poya.ParsedPage, text_hash: str, first_url_for_hash: dict[str, str]
) -> tuple[str, str | None]:
    if artifact.http_status != 200:
        return "skipped", f"http {artifact.http_status}"
    if artifact.source_url in poya.SKIP_URLS:
        return "skipped", poya.SKIP_URLS[artifact.source_url]
    if len(page.text) < MIN_TEXT_CHARS:
        return "skipped", "no text: content is in images (vision step not in v1)"
    first = first_url_for_hash.get(text_hash)
    if first and first != artifact.source_url:
        return "skipped", f"duplicate of {first}"
    return "active", None


def _list_date(meta: dict) -> date | None:
    value = meta.get("list_date")
    return date.fromisoformat(value) if value else None


def _delete_documents(session: Session) -> None:
    """Dev only: drop every document and everything built from it (chunks, promotions)."""
    for model in (Chunk, PromotionSource, Promotion, Document):
        session.execute(delete(model))
    session.flush()
