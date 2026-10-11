"""Step 4 (chunk): active documents → chunks of one corpus version (embeddings come in step 5).

A corpus version is "these inputs, indexed with these settings":
  fingerprint   hash of IndexConfig (chunk size, overlap, header, segmenter, word list, embed model)
  inputs_hash   hash of every document's content_hash plus the facts in its header line

Chunking is deterministic, so if a version with the same fingerprint and inputs_hash
already exists there is nothing to do. Change a setting, a page or a fact, and a new
version is built next to the old one. Versions that have embeddings stay (for A/B
comparison); drafts that were never embedded are replaced, since rebuilding them is free.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from promo_rag.db.models import Chunk, CorpusVersion, Document, Promotion, PromotionSource
from promo_rag.search.segment import segment
from promo_rag.workflows.ingestion.chunking import HeaderFacts, chunk_document
from promo_rag.workflows.ingestion.index_config import IndexConfig

SINGLE_DAY_TYPES = {"store_opening", "store_reopening"}


@dataclass
class ChunkReport:
    corpus_version_id: str = ""
    fingerprint: str = ""
    already_built: bool = False
    n_documents: int = 0
    n_chunks: int = 0
    by_doc_type: Counter = field(default_factory=Counter)
    multi_chunk_docs: int = 0
    token_counts: list[int] = field(default_factory=list)
    drafts_dropped: int = 0


def chunk_documents(session: Session, cfg: IndexConfig, rebuild: bool = False) -> ChunkReport:
    docs = session.scalars(
        select(Document)
        .where(Document.is_latest, Document.status == "active")
        .order_by(Document.source_url)
    ).all()
    facts = {doc.id: _header_facts(session, doc) for doc in docs}
    report = ChunkReport(fingerprint=cfg.fingerprint())

    inputs_hash = _inputs_hash(docs, facts)
    version = session.scalars(
        select(CorpusVersion).where(
            CorpusVersion.fingerprint == report.fingerprint,
            CorpusVersion.inputs_hash == inputs_hash,
        )
    ).one_or_none()
    if version and not rebuild:
        report.corpus_version_id, report.already_built = str(version.id), True
        report.n_documents, report.n_chunks = version.n_documents, version.n_chunks
        return report
    if version is None:
        version = CorpusVersion(
            fingerprint=report.fingerprint,
            config=cfg.model_dump(),
            inputs_hash=inputs_hash,
            status="building",
        )
        session.add(version)
        session.flush()
        report.drafts_dropped = _drop_unembedded_drafts(session, keep=version.id)
    else:
        session.execute(delete(Chunk).where(Chunk.corpus_version_id == version.id))

    for doc in docs:
        doc_facts, promotion = facts[doc.id]
        drafts = chunk_document(doc.full_text, doc_facts, cfg)
        for d in drafts:
            session.add(
                Chunk(
                    document_id=doc.id,
                    corpus_version_id=version.id,
                    chunk_index=d.index,
                    text=d.text,
                    search_text=segment(d.text, cfg.segmenter),
                    token_count=d.token_count,
                    doc_type=doc.doc_type,
                    language=doc.language,
                    start_date=promotion.start_date if promotion else None,
                    end_date=promotion.end_date if promotion else None,
                )
            )
            report.token_counts.append(d.token_count)
        report.by_doc_type[doc.doc_type] += len(drafts)
        report.multi_chunk_docs += len(drafts) > 1

    version.n_documents, version.n_chunks = len(docs), len(report.token_counts)
    session.commit()
    report.corpus_version_id = str(version.id)
    report.n_documents, report.n_chunks = version.n_documents, version.n_chunks
    return report


def _header_facts(session: Session, doc: Document) -> tuple[HeaderFacts, Promotion | None]:
    promotion = session.scalars(
        select(Promotion)
        .join(PromotionSource, PromotionSource.promotion_id == Promotion.id)
        .where(PromotionSource.document_id == doc.id)
    ).first()
    page = {"title": doc.title, "posted": doc.published_date, "section": doc.category}
    if promotion is None:
        return HeaderFacts(**page), None
    # stores is empty for "all stores", one name for a store notice, or the stores a
    # campaign is limited to; all three read right in the header.
    return (
        HeaderFacts(
            **page,
            start=promotion.start_date,
            end=promotion.end_date,
            stores=tuple(promotion.stores),
            single_day=promotion.promo_type in SINGLE_DAY_TYPES,
        ),
        promotion,
    )


def _inputs_hash(docs: list[Document], facts: dict) -> str:
    lines = sorted(f"{doc.content_hash}|{facts[doc.id][0]!r}" for doc in docs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _drop_unembedded_drafts(session: Session, keep) -> int:
    """Delete `building` versions with no embedded chunks: drafts replaced by this build."""
    has_embedding = exists().where(
        Chunk.corpus_version_id == CorpusVersion.id, Chunk.embedding.is_not(None)
    )
    drafts = session.scalars(
        select(CorpusVersion.id).where(
            CorpusVersion.status == "building", CorpusVersion.id != keep, ~has_embedding
        )
    ).all()
    for draft in drafts:
        session.execute(delete(Chunk).where(Chunk.corpus_version_id == draft))
        session.execute(delete(CorpusVersion).where(CorpusVersion.id == draft))
    return len(drafts)
