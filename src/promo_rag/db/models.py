"""Tables, in two Postgres schemas of one database.

platform: what any workflow uses
  datasets >──< artifacts                 (via dataset_artifacts)
  workflows 1──< workflow_runs 1──< action_runs
rag: what the RAG app reads and writes
  platform.artifacts 1──< documents >──< promotions      (via promotion_sources)
                             │  │
                             │  └──< extractions          (one row per extraction attempt)
                             └──< chunks >── corpus_versions
  embedding_cache: vectors keyed by (model, dims, sha256 of the input text)

Postgres holds all metadata; the blob store holds only raw bytes.
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
PLATFORM = {"schema": "platform"}
RAG = {"schema": "rag"}


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
    __table_args__ = (
        UniqueConstraint("source_url", "sha256"),
        Index("ix_artifacts_sha256", "sha256"),
        PLATFORM,
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    sha256: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str]
    final_url: Mapped[str | None]
    fetched_at: Mapped[datetime]  # first fetch that returned these bytes
    # Newest fetch that returned them. Parse picks each URL's newest by this, so a page
    # that changes A → B → A ends on A even though A's row is the older one.
    last_seen_at: Mapped[datetime]
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
        Index("ix_documents_content_hash", "content_hash"),
        RAG,
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("platform.artifacts.id"))
    source_url: Mapped[str]
    title: Mapped[str]
    language: Mapped[str] = mapped_column(server_default="zh")
    doc_type: Mapped[str]  # promotion | store_notice | announcement | faq_policy | catalog
    category: Mapped[str | None]  # the site's own category, e.g. 主題活動
    published_date: Mapped[date | None]
    content_hash: Mapped[str] = mapped_column(String(64))
    full_text: Mapped[str]
    image_urls: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    version: Mapped[int] = mapped_column(server_default="1")
    is_latest: Mapped[bool] = mapped_column(server_default="true")
    previous_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("rag.documents.id"))
    status: Mapped[str] = mapped_column(server_default="active")  # active | archived | skipped
    skip_reason: Mapped[str | None]  # e.g. "http 404", "duplicate of <id>"
    created_at: Mapped[datetime] = created_at()


class Promotion(Base):
    """Structured facts for one promotion. The SQL tools query this table."""

    __tablename__ = "promotions"
    __table_args__ = RAG

    id: Mapped[uuid.UUID] = uuid_pk()
    # Normalized title + dates. The same campaign on two pages gets one row.
    dedup_key: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str]
    # discount | bundle | gift | points | coupon | card_offer | store_opening | clearance_sale | other
    promo_type: Mapped[str]
    start_date: Mapped[date | None]
    end_date: Mapped[date | None]  # NULL = open-ended
    stores: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    # Full city or county names (高雄市, 新竹縣); store notices only, from the address.
    cities: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    channels: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    payment_methods: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    products: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}")
    min_spend: Mapped[Decimal | None]
    discount_text: Mapped[str]
    extraction_model: Mapped[str]
    extracted_at: Mapped[datetime]
    raw: Mapped[dict] = mapped_column(JSONB)  # the full Pydantic output
    created_at: Mapped[datetime] = created_at()


class Extraction(Base):
    """One fact-extraction run for one document: how it was done and how it went.

    Failures are recorded here and never become promotions rows.
    """

    __tablename__ = "extractions"
    __table_args__ = (UniqueConstraint("document_id", "schema_name", "prompt_version"), RAG)

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag.documents.id"))
    schema_name: Mapped[str]  # PromotionFacts | StoreNoticeFacts
    method: Mapped[str]  # rules | llm
    model: Mapped[str | None]  # NULL for rules
    prompt_version: Mapped[str]  # hash of prompt + schema (llm) or the rules version
    status: Mapped[str]  # ok | needs_review | failed
    output: Mapped[dict | None] = mapped_column(JSONB)  # the last output, valid or not
    errors: Mapped[list] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    attempts: Mapped[int]
    tokens_in: Mapped[int | None]
    tokens_out: Mapped[int | None]
    latency_ms: Mapped[int | None]
    created_at: Mapped[datetime] = created_at()


class PromotionSource(Base):
    """Which pages describe a promotion (usually one, sometimes two)."""

    __tablename__ = "promotion_sources"
    __table_args__ = RAG

    promotion_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag.promotions.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag.documents.id"), primary_key=True)


class CorpusVersion(Base):
    """One index build: these pages, indexed with these settings."""

    __tablename__ = "corpus_versions"
    __table_args__ = (
        Index("ix_corpus_versions_fingerprint", "fingerprint"),
        Index("uq_corpus_versions_fingerprint_inputs", "fingerprint", "inputs_hash", unique=True),
        # At most one active version: search reads it.
        Index(
            "uq_corpus_versions_one_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        RAG,
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    fingerprint: Mapped[str] = mapped_column(String(16))  # hash of `config`
    config: Mapped[dict] = mapped_column(JSONB)  # chunk size, overlap, embed model, dims, ...
    # Hash of everything the chunks were built from: each document's id, content_hash and
    # the facts in its header line. Same fingerprint + same inputs_hash = same chunks.
    inputs_hash: Mapped[str | None] = mapped_column(String(64))
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
        Index("ix_chunks_corpus_version_id", "corpus_version_id"),
        RAG,
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag.documents.id"))
    corpus_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag.corpus_versions.id"))
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


class EmbeddingCacheEntry(Base):
    """An embedding, keyed by the exact text that was sent (task prefix included)."""

    __tablename__ = "embedding_cache"
    __table_args__ = RAG

    model: Mapped[str] = mapped_column(primary_key=True)
    dims: Mapped[int] = mapped_column(primary_key=True)
    input_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())  # no fixed size
    created_at: Mapped[datetime] = created_at()


# ---------------------------------------------------------------- platform: workflows


class Dataset(Base):
    """A named group of artifacts: the input of a workflow run."""

    __tablename__ = "datasets"
    __table_args__ = PLATFORM

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str]
    description: Mapped[str | None]
    created_at: Mapped[datetime] = created_at()


class DatasetArtifact(Base):
    __tablename__ = "dataset_artifacts"
    __table_args__ = PLATFORM

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("platform.datasets.id", ondelete="CASCADE"), primary_key=True
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("platform.artifacts.id"), primary_key=True
    )


class Workflow(Base):
    """A workflow's runbook, synced from workflows/<name>/runbook.yaml when the app starts."""

    __tablename__ = "workflows"
    __table_args__ = PLATFORM

    name: Mapped[str] = mapped_column(primary_key=True)
    version: Mapped[int]
    description: Mapped[str | None]
    definition: Mapped[dict] = mapped_column(JSONB)  # the runbook, as parsed
    runbook_hash: Mapped[str] = mapped_column(String(64))
    synced_at: Mapped[datetime] = mapped_column(server_default=func.now())


class WorkflowRun(Base):
    """One run of a workflow. A `queued` row is the job: a worker claims it."""

    __tablename__ = "workflow_runs"
    __table_args__ = (
        Index(
            "ix_workflow_runs_queued",
            "created_at",
            postgresql_where=text("status = 'queued'"),
        ),
        PLATFORM,
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    workflow_name: Mapped[str] = mapped_column(ForeignKey("platform.workflows.name"))
    workflow_version: Mapped[int]
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("platform.datasets.id"))
    params: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(
        server_default="queued"
    )  # queued | running | succeeded | failed
    output: Mapped[dict | None] = mapped_column(JSONB)  # the run report (GET /runs/{id}/content)
    error: Mapped[str | None]
    worker: Mapped[str | None]  # host:pid of the worker that claimed it
    created_at: Mapped[datetime] = created_at()
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class ActionRun(Base):
    """One action (step) of a workflow run."""

    __tablename__ = "action_runs"
    __table_args__ = (UniqueConstraint("run_id", "position"), PLATFORM)

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("platform.workflow_runs.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int]
    name: Mapped[str]
    status: Mapped[str] = mapped_column(
        server_default="pending"
    )  # pending | running | succeeded | failed | skipped
    output: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
