"""Step 5 (embed): fill chunks.embedding for one corpus version.

Uses the version's own stored settings (model, dims, input format), not today's .env,
so every vector in a version matches its fingerprint. Commits after each batch: an
interrupted run picks up the chunks that are still empty.
"""

import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from promo_rag.config import get_settings
from promo_rag.db.models import Chunk, CorpusVersion, Document
from promo_rag.llm import Pacer, cost_usd
from promo_rag.search.embed import (
    DbEmbeddingCache,
    EmbedCall,
    document_input,
    embed_inputs,
    gemini_embed,
    input_hash,
    norm,
)
from promo_rag.workflows.ingestion.index_config import IndexConfig


@dataclass
class EmbedReport:
    corpus_version_id: str = ""
    model: str = ""
    dims: int = 0
    embedded: int = 0
    from_cache: int = 0
    from_api: int = 0
    api_calls: int = 0
    estimated_tokens: int = 0
    seconds: float = 0.0
    norms: list[float] = field(default_factory=list)

    @property
    def estimated_cost_usd(self) -> float:
        return cost_usd(self.model, self.estimated_tokens, 0)


def embed_chunks(
    session: Session,
    version_id: uuid.UUID | None = None,
    batch_size: int = 50,
    call: EmbedCall | None = None,
    log=print,
) -> EmbedReport:
    version = _target_version(session, version_id)
    cfg = IndexConfig(**version.config)
    call = call or gemini_embed(cfg.embed_model, cfg.embed_dims)
    pacer = Pacer(get_settings().gemini_rpm)
    cache = DbEmbeddingCache(session)
    report = EmbedReport(str(version.id), cfg.embed_model, cfg.embed_dims)

    todo = session.execute(
        select(Chunk.id, Chunk.text, Chunk.token_count, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.corpus_version_id == version.id, Chunk.embedding.is_(None))
        .order_by(Document.source_url, Chunk.chunk_index)
    ).all()
    started = time.monotonic()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        inputs = [document_input(cfg.embed_doc_format, row.title, row.text) for row in batch]
        result = embed_inputs(
            inputs,
            model=cfg.embed_model,
            dims=cfg.embed_dims,
            cache=cache,
            call=call,
            pacer=pacer,
            batch_size=batch_size,
        )
        for row, vector in zip(batch, result.vectors, strict=True):
            session.execute(
                update(Chunk)
                .where(Chunk.id == row.id)
                .values(embedding=vector, embed_model=cfg.embed_model)
            )
            report.norms.append(norm(vector))
        session.commit()

        report.embedded += len(batch)
        report.from_cache += result.from_cache
        report.from_api += result.from_api
        report.api_calls += result.api_calls
        report.estimated_tokens += sum(
            row.token_count
            for row, text in zip(batch, inputs, strict=True)
            if input_hash(text) in result.fetched
        )
        log(f"  {report.embedded}/{len(todo)} chunks ({result.from_api} from the API)")
    report.seconds = time.monotonic() - started
    return report


def _target_version(session: Session, version_id: uuid.UUID | None) -> CorpusVersion:
    """The given version, or else the newest one still being built."""
    query = select(CorpusVersion)
    if version_id:
        query = query.where(CorpusVersion.id == version_id)
    else:
        query = query.where(CorpusVersion.status == "building").order_by(
            CorpusVersion.created_at.desc()
        )
    version = session.scalars(query).first()
    if version is None:
        raise LookupError("no corpus version to embed; run `ingest chunk` first")
    return version
