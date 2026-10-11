"""Step 6 (activate): check a corpus version and make it the one search reads.

Every check must pass, or nothing changes:
  - every chunk has an embedding, from the version's model, with the version's dims
  - every active document has at least one chunk
  - the documents and facts haven't changed since the chunks were built (inputs_hash)
Then, in one transaction, the previous active version is retired and this one becomes
active (the database allows only one). Finally Postgres's statistics are refreshed and the
vector (HNSW) and keyword (GIN) indexes are checked: valid, and usable by a search query.
"""

import json
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from promo_rag.db.models import Chunk, CorpusVersion, Document
from promo_rag.workflows.ingestion.actions.chunk import header_facts, inputs_hash
from promo_rag.workflows.ingestion.index_config import IndexConfig

INDEXES = ("ix_chunks_embedding_hnsw", "ix_chunks_tsv")


@dataclass
class VersionStats:
    n_chunks: int
    missing_embeddings: int
    wrong_model: int
    wrong_dims: int
    docs_without_chunks: list[str]
    stale: bool


def problems(stats: VersionStats) -> list[str]:
    """Why this version can't be activated. Empty list = ready."""
    out = []
    if stats.n_chunks == 0:
        out.append("the version has no chunks")
    if stats.missing_embeddings:
        out.append(f"{stats.missing_embeddings} chunks have no embedding: run `ingest embed`")
    if stats.wrong_model:
        out.append(f"{stats.wrong_model} chunks were embedded with a different model")
    if stats.wrong_dims:
        out.append(f"{stats.wrong_dims} embeddings have the wrong number of dims")
    if stats.docs_without_chunks:
        out.append(
            f"{len(stats.docs_without_chunks)} active pages have no chunks, e.g. "
            f"{stats.docs_without_chunks[0]}: run `ingest chunk`"
        )
    if stats.stale:
        out.append(
            "pages or extracted facts changed after this version was built: run `ingest chunk`"
        )
    return out


@dataclass
class ActivateReport:
    version_id: str = ""
    fingerprint: str = ""
    n_chunks: int = 0
    problems: list[str] = field(default_factory=list)
    activated: bool = False
    already_active: bool = False
    retired: str | None = None
    indexes_valid: dict[str, bool] = field(default_factory=dict)
    plans: dict[str, str] = field(default_factory=dict)  # query → how Postgres runs it


def activate_version(session: Session, version_id: uuid.UUID | None = None) -> ActivateReport:
    version = _target_version(session, version_id)
    cfg = IndexConfig(**version.config)
    report = ActivateReport(str(version.id), version.fingerprint, version.n_chunks)

    if version.status == "active":
        report.already_active = True
    else:
        report.problems = problems(_stats(session, version, cfg))
        if report.problems:
            return report
        previous = session.scalars(
            select(CorpusVersion).where(CorpusVersion.status == "active")
        ).one_or_none()
        if previous is not None:
            previous.status = "retired"
            report.retired = str(previous.id)
            session.flush()  # the one-active rule is checked per statement
        version.status = "active"
        session.commit()
        report.activated = True

    session.execute(text("ANALYZE chunks"))
    session.commit()
    report.indexes_valid = _indexes_valid(session)
    report.plans = _plans(session, version.id)
    return report


def _target_version(session: Session, version_id: uuid.UUID | None) -> CorpusVersion:
    """The given version, or else the newest one."""
    query = select(CorpusVersion)
    query = (
        query.where(CorpusVersion.id == version_id)
        if version_id
        else query.order_by(CorpusVersion.created_at.desc())
    )
    version = session.scalars(query).first()
    if version is None:
        raise LookupError("no corpus version; run `ingest chunk` and `ingest embed` first")
    return version


def _stats(session: Session, version: CorpusVersion, cfg: IndexConfig) -> VersionStats:
    in_version = Chunk.corpus_version_id == version.id
    has_vec = Chunk.embedding.is_not(None)

    def count(*conditions) -> int:
        return session.scalar(
            select(func.count()).select_from(Chunk).where(in_version, *conditions)
        )

    docs = session.scalars(
        select(Document)
        .where(Document.is_latest, Document.status == "active")
        .order_by(Document.source_url)
    ).all()
    with_chunks = set(session.scalars(select(Chunk.document_id).where(in_version).distinct()))
    facts = {doc.id: header_facts(session, doc) for doc in docs}
    return VersionStats(
        n_chunks=count(),
        missing_embeddings=count(~has_vec),
        wrong_model=count(has_vec, Chunk.embed_model.is_distinct_from(cfg.embed_model)),
        wrong_dims=count(has_vec, func.vector_dims(Chunk.embedding) != cfg.embed_dims),
        docs_without_chunks=[d.source_url for d in docs if d.id not in with_chunks],
        stale=inputs_hash(docs, facts) != version.inputs_hash,
    )


def _indexes_valid(session: Session) -> dict[str, bool]:
    rows = session.execute(
        text(
            "SELECT c.relname, i.indisvalid FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = ANY(:names)"
        ),
        {"names": list(INDEXES)},
    ).all()
    found = dict(rows)
    return {name: found.get(name, False) for name in INDEXES}


def _plans(session: Session, version_id: uuid.UUID) -> dict[str, str]:
    """How Postgres runs the searches, from EXPLAIN.

    On a small table, filtering to the version and sorting exactly by distance is cheaper
    than the HNSW index, so search won't use HNSW yet. The second check proves the HNSW
    index is usable for when the table grows.
    """
    vector = session.scalar(
        select(Chunk.embedding).where(Chunk.corpus_version_id == version_id).limit(1)
    )
    q = {"v": version_id, "q": str(list(vector))}
    checks = {
        "vector search (as search runs it)": (
            [],
            (
                "SELECT id FROM chunks WHERE corpus_version_id = :v "
                "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 20"
            ),
        ),
        "vector index usable (HNSW)": (
            ["SET LOCAL enable_seqscan = off"],
            "SELECT id FROM chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT 20",
        ),
        "keyword search (GIN)": (
            [],
            (
                "SELECT id FROM chunks WHERE corpus_version_id = :v "
                "AND tsv @@ to_tsquery('simple', '回饋金') LIMIT 20"
            ),
        ),
    }
    plans = {}
    for name, (settings, sql) in checks.items():
        for setting in settings:
            session.execute(text(setting))
        plan = session.scalar(text(f"EXPLAIN (FORMAT JSON) {sql}"), q)
        plans[name] = _scan_nodes(plan if isinstance(plan, list) else json.loads(plan))
        session.rollback()  # drop any SET LOCAL
    return plans


def _scan_nodes(plan) -> str:
    """The scan steps of an EXPLAIN plan, e.g. 'Index Scan on ix_chunks_embedding_hnsw'."""
    out = []

    def walk(node):
        if "Scan" in node.get("Node Type", ""):
            target = node.get("Index Name") or node.get("Relation Name")
            out.append(f"{node['Node Type']} on {target}")
        for child in node.get("Plans", []):
            walk(child)

    walk(plan[0]["Plan"])
    return ", ".join(out)
