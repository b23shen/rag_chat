"""Ingestion tables. Postgres holds all metadata; the blob store holds only raw bytes.

artifacts 1──< documents >──< promotions      (via promotion_sources)
                   │
                   └──< chunks >── corpus_versions
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIMS = 1536


class Base(DeclarativeBase):
    # Every datetime column is timestamptz.
    type_annotation_map: ClassVar[dict[Any, Any]] = {datetime: DateTime(timezone=True)}


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))


def created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


class Artifact(Base):
    """One fetch: the exact bytes one URL returned at one time."""

    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("source_url", "sha256"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str]
    final_url: Mapped[str | None]
    fetched_at: Mapped[datetime]
    http_status: Mapped[int]
    mime_type: Mapped[str | None]
    byte_size: Mapped[int]
    # Path in the blob store. Derived from sha256, so identical bytes are stored once.
    blob_key: Mapped[str]
    # Whatever the crawler knew from the list pages (category, list date, list title).
    fetch_meta: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = created_at()


class Document(Base):
    """One parsed version of a page."""

    __tablename__ = "documents"
    __table_args__ = (
        # Only one latest version per URL.
        Index(
            "uq_documents_latest_url",
            "source_url",
            unique=True,
            postgresql_where=text("is_latest"),
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id"))
    source_url: Mapped[str]
    title: Mapped[str]
    language: Mapped[str] = mapped_column(server_default="zh")
    doc_type: Mapped[str]  # promotion | store_notice | announcement | faq_policy | catalog
    category: Mapped[str | None]  # the site's own category, e.g. 主題活動
    published_date: Mapped[date | None]
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    full_text: Mapped[str]
    image_urls: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    version: Mapped[int] = mapped_column(server_default="1")
    is_latest: Mapped[bool] = mapped_column(server_default="true")
    previous_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id"))
    status: Mapped[str] = mapped_column(server_default="active")  # active | archived | skipped
    skip_reason: Mapped[str | None]  # e.g. "http 404", "duplicate of <id>"
    created_at: Mapped[datetime] = created_at()


class Promotion(Base):
    """Structured facts for one promotion. The SQL tools query this table."""

    __tablename__ = "promotions"

    id: Mapped[uuid.UUID] = uuid_pk()
    # Normalized title + dates. The same campaign on two pages gets one row.
    dedup_key: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str]
    # discount | bundle | gift | points | coupon | card_offer | store_opening | clearance_sale | other
    promo_type: Mapped[str]
    start_date: Mapped[date | None]
    end_date: Mapped[date | None]  # NULL = open-ended
    stores: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    channels: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    payment_methods: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    products: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    min_spend: Mapped[Decimal | None]
    discount_text: Mapped[str]
    extraction_model: Mapped[str]
    extracted_at: Mapped[datetime]
    raw: Mapped[dict] = mapped_column(JSONB)  # the full Pydantic output
    created_at: Mapped[datetime] = created_at()


class PromotionSource(Base):
    """Which pages describe a promotion (usually one, sometimes two)."""

    __tablename__ = "promotion_sources"

    promotion_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("promotions.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), primary_key=True)


class CorpusVersion(Base):
    """One index build: these pages, indexed with these settings."""

    __tablename__ = "corpus_versions"

    id: Mapped[uuid.UUID] = uuid_pk()
    fingerprint: Mapped[str] = mapped_column(String(16), index=True)  # hash of `config`
    config: Mapped[dict] = mapped_column(JSONB)  # chunk size, overlap, embed model, dims, ...
    artifact_set_hash: Mapped[str | None] = mapped_column(String(64))
    n_documents: Mapped[int] = mapped_column(server_default="0")
    n_chunks: Mapped[int] = mapped_column(server_default="0")
    status: Mapped[str] = mapped_column(server_default="building")  # building | active | retired
    created_at: Mapped[datetime] = created_at()


class Chunk(Base):
    """A search unit."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "corpus_version_id", "chunk_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    corpus_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("corpus_versions.id"), index=True
    )
    chunk_index: Mapped[int]
    text: Mapped[str]  # what the LLM reads (header line + body)
    search_text: Mapped[str]  # jieba-segmented text for keyword search
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple', search_text)", persisted=True)
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIMS))
    embed_model: Mapped[str | None]
    token_count: Mapped[int]
    # Copied from the document/promotion so filters run on this table alone.
    doc_type: Mapped[str]
    language: Mapped[str]
    start_date: Mapped[date | None]
    end_date: Mapped[date | None]
    created_at: Mapped[datetime] = created_at()
