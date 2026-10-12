"""The ingestion workflow's actions, under the names its runbook uses.

Each takes an ActionContext and returns a small JSON report, stored in action_runs.output
and shown by GET /runs/{id}/content. The same functions back the `ingest <step>` commands.
"""

import uuid

from promo_rag.platform.datasets import dataset_artifact_ids
from promo_rag.platform.runbook import ActionContext, ActionFailed
from promo_rag.workflows.ingestion import deps
from promo_rag.workflows.ingestion.actions.activate import activate_version
from promo_rag.workflows.ingestion.actions.chunk import chunk_documents
from promo_rag.workflows.ingestion.actions.embed import embed_chunks
from promo_rag.workflows.ingestion.actions.extract import extract_documents
from promo_rag.workflows.ingestion.actions.parse import parse_documents
from promo_rag.workflows.ingestion.index_config import IndexConfig

INDEX_PARAMS = ("chunk_tokens", "overlap_tokens", "segmenter", "header")


def parse(ctx: ActionContext) -> dict:
    artifacts = (
        dataset_artifact_ids(ctx.session, uuid.UUID(ctx.dataset_id)) if ctx.dataset_id else None
    )
    r = parse_documents(ctx.session, deps.blob_store(), artifact_ids=artifacts)
    return {
        "added": r.added,
        "new_versions": r.new_versions,
        "unchanged": r.unchanged,
        "promotions_removed": r.promotions_removed,
        "by_status": dict(r.by_status),
        "skipped": [{"url": url, "reason": reason} for url, reason in r.skipped],
    }


def extract(ctx: ActionContext) -> dict:
    r = extract_documents(ctx.session, log=ctx.log)
    return {
        "extraction_run": r.run_id,
        "by_status": dict(r.by_status),
        "by_method": dict(r.by_method),
        "already_done": r.skipped_done,
        "promotions_new": r.promotions_new,
        "promotions_merged": r.promotions_merged,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "cost_usd": round(r.cost_usd, 4),
        "problems": [{"url": u, "status": s, "errors": e} for u, s, e in r.problems],
    }


def chunk(ctx: ActionContext) -> dict:
    cfg = IndexConfig(**{k: ctx.params[k] for k in INDEX_PARAMS if k in ctx.params})
    r = chunk_documents(ctx.session, cfg)
    return {
        "corpus_version_id": r.corpus_version_id,
        "fingerprint": r.fingerprint,
        "already_built": r.already_built,
        "documents": r.n_documents,
        "chunks": r.n_chunks,
        "drafts_dropped": r.drafts_dropped,
    }


def embed(ctx: ActionContext) -> dict:
    version_id = uuid.UUID(ctx.outputs["chunk"]["corpus_version_id"])
    r = embed_chunks(ctx.session, version_id=version_id, log=ctx.log)
    return {
        "corpus_version_id": r.corpus_version_id,
        "embedded": r.embedded,
        "from_cache": r.from_cache,
        "from_api": r.from_api,
        "api_calls": r.api_calls,
        "estimated_tokens": r.estimated_tokens,
        "estimated_cost_usd": round(r.estimated_cost_usd, 4),
    }


def activate(ctx: ActionContext) -> dict:
    version_id = uuid.UUID(ctx.outputs["chunk"]["corpus_version_id"])
    r = activate_version(ctx.session, version_id)
    if r.problems:
        raise ActionFailed("not activated: " + "; ".join(r.problems))
    return {
        "corpus_version_id": r.version_id,
        "activated": r.activated,
        "already_active": r.already_active,
        "retired": r.retired,
        "indexes_valid": r.indexes_valid,
        "plans": r.plans,
    }


ACTIONS = {
    "parse": parse,
    "extract": extract,
    "chunk": chunk,
    "embed": embed,
    "activate": activate,
}
