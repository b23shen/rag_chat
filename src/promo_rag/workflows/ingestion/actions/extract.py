"""Step 3 (extract): documents → promotions, through checked facts.

For each active document whose doc_type has an extractor (facts.EXTRACTORS):
  1. Rules first, when the format is fixed (store notices). Otherwise, or if the rules
     don't match, the `fact_extractor` agent fills the Pydantic schema for that doc_type.
  2. Check the facts in code (facts.check_facts). If a check fails, send the errors back
     to the agent once and ask again.
  3. Log the run in `extractions` (ok | needs_review | failed).
  4. Only if ok: upsert the promotions row by dedup_key and link the page in
     promotion_sources. The same campaign on two pages ends up as one promotion.

A failed extraction never blocks search: chunks are built from the document text either way.
"""

import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from promo_rag.agents import AgentError, AgentOutputError, TraceInfo, invoke_agent
from promo_rag.agents.config import get_agent_config
from promo_rag.agents.invoke import prompt_version
from promo_rag.config import get_settings
from promo_rag.db.models import Document, Extraction, Promotion, PromotionSource
from promo_rag.llm import Pacer, cost_usd
from promo_rag.workflows.ingestion.facts import (
    EXTRACTORS,
    RULES_VERSION,
    Extractor,
    check_facts,
    dedup_key,
)

AGENT = "fact_extractor"
MAX_ATTEMPTS = 2  # the first answer + one retry with the check errors


@dataclass
class Outcome:
    method: str
    status: str
    facts: BaseModel | None = None
    output: dict | None = None
    errors: list[list[str]] = field(default_factory=list)
    attempts: int = 0
    model: str | None = None
    prompt_version: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


@dataclass
class ExtractReport:
    run_id: str = ""
    by_status: Counter = field(default_factory=Counter)
    by_method: Counter = field(default_factory=Counter)
    skipped_done: int = 0
    promotions_new: int = 0
    promotions_merged: int = 0  # a second page linked to an existing promotion
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    problems: list[tuple[str, str, list]] = field(default_factory=list)  # (url, status, errors)


def extract_documents(
    session: Session,
    force: bool = False,
    limit: int | None = None,
    only: str | None = None,
    log=print,
) -> ExtractReport:
    pacer = Pacer(get_settings().gemini_rpm)
    report = ExtractReport(run_id=f"extract-{uuid.uuid4().hex[:8]}")
    agent_config = get_agent_config(AGENT)

    docs = session.scalars(
        select(Document)
        .where(
            Document.is_latest,
            Document.status == "active",
            Document.doc_type.in_(list(EXTRACTORS)),
        )
        .order_by(Document.doc_type, Document.source_url)
    ).all()
    if only:
        docs = [d for d in docs if only in d.source_url]
    if limit:
        docs = docs[:limit]

    for i, doc in enumerate(docs, 1):
        extractor = EXTRACTORS[doc.doc_type]
        schema_name = extractor.schema.__name__
        llm_version = prompt_version(agent_config, extractor.schema)
        done = session.scalars(
            select(Extraction).where(
                Extraction.document_id == doc.id,
                Extraction.schema_name == schema_name,
                Extraction.status == "ok",
                Extraction.prompt_version.in_([RULES_VERSION, llm_version]),
            )
        ).first()
        if done and not force:
            report.skipped_done += 1
            continue

        _unlink_document(session, doc)
        trace = TraceInfo(
            workflow="ingestion",
            run_id=report.run_id,
            metadata={"document_id": str(doc.id), "url": doc.source_url},
        )
        outcome = _extract_one(doc, extractor, trace, pacer)
        session.add(
            Extraction(
                document_id=doc.id,
                schema_name=schema_name,
                method=outcome.method,
                model=outcome.model,
                prompt_version=outcome.prompt_version or llm_version,
                status=outcome.status,
                output=outcome.output,
                errors=outcome.errors,
                attempts=outcome.attempts,
                tokens_in=outcome.tokens_in or None,
                tokens_out=outcome.tokens_out or None,
                latency_ms=outcome.latency_ms or None,
            )
        )
        if outcome.status == "ok":
            merged = _upsert_promotion(session, doc, extractor, outcome)
            report.promotions_merged += merged
            report.promotions_new += not merged
        else:
            report.problems.append((doc.source_url, outcome.status, outcome.errors))
        session.commit()

        report.by_status[outcome.status] += 1
        report.by_method[outcome.method] += 1
        report.tokens_in += outcome.tokens_in
        report.tokens_out += outcome.tokens_out
        if outcome.model:
            report.cost_usd += cost_usd(outcome.model, outcome.tokens_in, outcome.tokens_out)
        log(
            f"[{i}/{len(docs)}] {outcome.status:12s} {outcome.method:5s} "
            f"{doc.source_url.removeprefix('https://www.poya.com.tw')}"
            + (
                f"  ({outcome.latency_ms} ms, {outcome.attempts} attempt(s))"
                if outcome.model
                else ""
            )
        )
    return report


def _shown_text(doc: Document) -> str:
    """Everything the extractor sees. Checks must search all of it: numbers are often in the title."""
    return f"{doc.title}\n{doc.full_text}"


def _extract_one(doc: Document, extractor: Extractor, trace: TraceInfo, pacer: Pacer) -> Outcome:
    if extractor.rules is not None:
        facts = extractor.rules(doc.title, doc.full_text, doc.published_date)
        if facts is not None:
            errors = check_facts(facts, _shown_text(doc), doc.published_date)
            return Outcome(
                method="rules",
                status="ok" if not errors else "needs_review",
                facts=facts,
                output=facts.model_dump(mode="json"),
                errors=[errors] if errors else [],
                attempts=1,
                prompt_version=RULES_VERSION,
            )
    return extract_with_agent(doc, extractor.schema, trace, pacer)


def extract_with_agent(
    doc: Document,
    schema: type[BaseModel],
    trace: TraceInfo,
    pacer: Pacer,
    invoke: Callable = invoke_agent,
) -> Outcome:
    """Ask the fact_extractor agent; check its answer; on failed checks, send the errors back once."""
    outcome = Outcome(method="llm", status="needs_review")
    published = doc.published_date.isoformat() if doc.published_date else "unknown"
    messages: list[BaseMessage] = [
        HumanMessage(
            f"Title: {doc.title}\nCategory: {doc.category}\nPublished: {published}\n"
            f"<page>\n{doc.full_text}\n</page>"
        )
    ]
    for _ in range(MAX_ATTEMPTS):
        outcome.attempts += 1
        pacer.wait()
        try:
            result = invoke(
                AGENT,
                messages,
                variables={"published_date": published},
                response_schema=schema,
                trace=trace,
            )
        except AgentOutputError as e:
            outcome.errors.append([f"The answer doesn't match the schema: {e.cause}"])
            messages = [
                *messages,
                HumanMessage("Your answer wasn't valid for the schema. Answer again."),
            ]
            continue
        except AgentError as e:
            outcome.status = "failed"
            outcome.errors.append([str(e)])
            return outcome

        outcome.model, outcome.prompt_version = result.model, result.prompt_version
        outcome.tokens_in += result.tokens_in
        outcome.tokens_out += result.tokens_out
        outcome.latency_ms += result.latency_ms
        facts = result.output
        outcome.output = facts.model_dump(mode="json")
        errors = check_facts(facts, _shown_text(doc), doc.published_date)
        if not errors:
            outcome.status, outcome.facts = "ok", facts
            return outcome

        outcome.errors.append(errors)
        messages = [
            *result.messages,
            HumanMessage(
                "Your answer failed these checks:\n- "
                + "\n- ".join(errors)
                + "\nFix them and answer again."
            ),
        ]
    return outcome


def _upsert_promotion(
    session: Session, doc: Document, extractor: Extractor, outcome: Outcome
) -> bool:
    """Insert the promotion, or link this page to the same promotion found on another page.

    Returns True if the promotion already existed (a merge).
    """
    row = extractor.to_promotion(outcome.facts, doc.title)
    key = dedup_key(row["title"], row["start_date"], row["end_date"])
    new_id = session.execute(
        insert(Promotion)
        .values(
            **row,
            dedup_key=key,
            extraction_model=outcome.model or outcome.prompt_version,
            extracted_at=datetime.now(UTC),
            raw=outcome.output,
        )
        .on_conflict_do_nothing(index_elements=["dedup_key"])
        .returning(Promotion.id)
    ).scalar_one_or_none()
    promotion_id = (
        new_id or session.scalars(select(Promotion.id).where(Promotion.dedup_key == key)).one()
    )
    session.execute(
        insert(PromotionSource)
        .values(promotion_id=promotion_id, document_id=doc.id)
        .on_conflict_do_nothing()
    )
    return new_id is None


def _unlink_document(session: Session, doc: Document) -> None:
    """Before re-extracting: drop this page's old runs and links, and promotions left with no page."""
    old = session.scalars(
        select(PromotionSource.promotion_id).where(PromotionSource.document_id == doc.id)
    ).all()
    session.execute(delete(PromotionSource).where(PromotionSource.document_id == doc.id))
    session.execute(delete(Extraction).where(Extraction.document_id == doc.id))
    for promotion_id in old:
        still_linked = session.scalars(
            select(PromotionSource).where(PromotionSource.promotion_id == promotion_id)
        ).first()
        if still_linked is None:
            session.execute(delete(Promotion).where(Promotion.id == promotion_id))
