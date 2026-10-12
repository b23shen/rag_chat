"""Datasets and artifacts: what a workflow run takes as input.

create_dataset   a named, empty group
add_artifact     store the bytes (blob key = sha256) and record the fetch; the same
                 (URL, sha256) again is the same artifact
link_artifacts   put artifacts in a dataset
"""

import hashlib
import uuid
from datetime import datetime

from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from promo_rag.db.models import Artifact, Dataset, DatasetArtifact
from promo_rag.storage.blob import BlobStore


def create_dataset(session: Session, name: str, description: str | None = None) -> Dataset:
    dataset = Dataset(name=name, description=description)
    session.add(dataset)
    session.flush()
    return dataset


def add_artifact(
    session: Session,
    store: BlobStore,
    data: bytes,
    *,
    source_url: str,
    fetched_at: datetime,
    http_status: int = 200,
    final_url: str | None = None,
    mime_type: str = "text/html",
    fetch_meta: dict | None = None,
) -> tuple[uuid.UUID, bool]:
    """Returns (artifact_id, created). Safe to repeat."""
    sha = hashlib.sha256(data).hexdigest()
    key = store.put(data, ".html" if mime_type == "text/html" else "")
    stmt = insert(Artifact).values(
        sha256=sha,
        source_url=source_url,
        final_url=final_url,
        fetched_at=fetched_at,
        last_seen_at=fetched_at,
        http_status=http_status,
        mime_type=mime_type,
        byte_size=len(data),
        blob_key=key,
        fetch_meta=fetch_meta or {},
    )
    # Seen before: keep the row, move last_seen_at forward. (xmax = 0) is true only for a
    # row this statement inserted, so it tells new from existing in one round trip.
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_url", "sha256"],
        set_={"last_seen_at": func.greatest(Artifact.last_seen_at, stmt.excluded.last_seen_at)},
    ).returning(Artifact.id, literal_column("(xmax = 0)"))
    artifact_id, created = session.execute(stmt).one()
    return artifact_id, created


def link_artifacts(session: Session, dataset_id: uuid.UUID, artifact_ids: list[uuid.UUID]) -> None:
    if artifact_ids:
        session.execute(
            insert(DatasetArtifact)
            .values([{"dataset_id": dataset_id, "artifact_id": a} for a in artifact_ids])
            .on_conflict_do_nothing()
        )


def dataset_artifact_ids(session: Session, dataset_id: uuid.UUID) -> list[uuid.UUID]:
    return list(
        session.scalars(
            select(DatasetArtifact.artifact_id).where(DatasetArtifact.dataset_id == dataset_id)
        )
    )


def dataset_size(session: Session, dataset_id: uuid.UUID) -> int:
    return session.scalar(
        select(func.count())
        .select_from(DatasetArtifact)
        .where(DatasetArtifact.dataset_id == dataset_id)
    )
