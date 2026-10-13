"""SQL tools over rag.promotions: count and list promotions by type, date, store, city,
payment method or channel, and get one promotion with its pages.

Search ranks chunks by topic, so it can't answer "how many stores opened in August?": it
returns the top 5, and it doesn't check dates (M2 handoff). These tools filter the
promotions table instead, so every match is found and counted.

Plain functions that take a Session, so tests and the MCP server (M5) call them directly;
tools/__init__.py wraps them as LangChain tools. The model only fills typed filters; it
never writes SQL.

Dates. A store opening or reopening is one day (its start_date; end_date is NULL). Any other
promotion runs from start_date to end_date, where a NULL end_date means open-ended and a NULL
start_date means "already running" (the page gives no start).
"""

import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, String, case, cast, func, or_, select
from sqlalchemy.orm import Session

from promo_rag.db.models import Document, Promotion, PromotionSource
from promo_rag.workflows.ingestion.facts import CITIES, SINGLE_DAY_TYPES

PromoType = Literal[
    "discount", "cashback", "points", "coupon", "gift", "card_offer", "lottery",
    "store_opening", "store_reopening", "clearance_sale", "other",
]  # fmt: skip
MAX_LIMIT = 50
PAGE_CHARS = 4000  # campaign pages average about 900 characters; the longest is about 3,800
# Facts kept only in `raw` (they belong to one page type) that are still worth showing.
RAW_DETAILS = ("period_text", "excluded_stores", "opening_time", "address", "phone")


class PromotionFilters(BaseModel):
    """Every filter is optional; the filters given must all match."""

    promo_types: list[PromoType] | None = Field(
        default=None,
        description="Kinds of promotion. Store events: store_opening (a new store), "
        "store_reopening (after renovation), clearance_sale (before renovation). Offers: "
        "discount, cashback, points, coupon, gift, card_offer, lottery, other.",
    )
    active_from: date | None = Field(
        default=None,
        description="Promotions still running on or after this day. With active_to, the "
        "promotions running at any time in that period (e.g. this month, or today: both "
        "the same day).",
    )
    active_to: date | None = Field(
        default=None, description="Promotions already started on or before this day."
    )
    start_from: date | None = Field(
        default=None,
        description="Promotions that start (or stores that open) on or after this day. "
        "Use start_from/start_to for 'opened in August' or 'started in May'.",
    )
    start_to: date | None = Field(
        default=None, description="Promotions that start on or before this day."
    )
    store: str | None = Field(
        default=None,
        description="Part of a store name, e.g. 新店中正 or 新店中正店. Matches promotions that "
        "name the store; offers for all stores name none.",
    )
    city: Literal[CITIES] | None = Field(
        default=None,
        description="A city or county, as its full Chinese name (Kaohsiung = 高雄市, "
        "Taichung = 台中市). Only store events have a city.",
    )
    payment_method: str | None = Field(
        default=None,
        description="Part of a payment method's name as pages write it, e.g. LINE Pay, "
        "街口, POYA支付, 信用卡 (matches 台新銀行信用卡, not 美國運通卡).",
    )
    channel: Literal["store", "online", "app"] | None = Field(
        default=None, description="Where it applies: store, online (the online shop), app."
    )


class ListPromotionsArgs(PromotionFilters):
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT, description="At most this many rows.")


def count_promotions(session: Session, filters: PromotionFilters) -> dict:
    """{"count": n, "by_type": {promo_type: n}} for the promotions that match."""
    rows = session.execute(
        select(Promotion.promo_type, func.count())
        .where(*_conditions(filters))
        .group_by(Promotion.promo_type)
        .order_by(func.count().desc(), Promotion.promo_type)
    ).all()
    by_type = {promo_type: n for promo_type, n in rows}
    return {"count": sum(by_type.values()), "by_type": by_type}


def list_promotions(session: Session, args: ListPromotionsArgs) -> dict:
    """The matching promotions, earliest start first, with the pages they come from."""
    conditions = _conditions(args)
    total = session.scalar(select(func.count()).select_from(Promotion).where(*conditions))
    promotions = session.scalars(
        select(Promotion)
        .where(*conditions)
        .order_by(Promotion.start_date.asc().nulls_first(), Promotion.title)
        .limit(args.limit)
    ).all()
    urls = _source_urls(session, [p.id for p in promotions])
    return {
        "total": total,
        "shown": len(promotions),
        "promotions": [_row(p, urls.get(p.id, [])) for p in promotions],
    }


class GetPromotionArgs(BaseModel):
    id: str = Field(
        min_length=8,
        description="The promotion's id from list_promotions (the first 8 characters are enough).",
    )


def get_promotion(session: Session, args: GetPromotionArgs) -> dict:
    """One promotion: every fact, and each page it comes from with the page's text.

    Returns {"error": ...} if the id matches no promotion, or more than one.
    """
    prefix = args.id.strip().lower()
    matches = session.scalars(
        select(Promotion)
        .where(cast(Promotion.id, String).startswith(prefix, autoescape=True))
        .limit(2)
    ).all()
    if len(matches) != 1:
        problem = "no promotion has" if not matches else "more than one promotion has"
        return {"error": f"{problem} an id starting with {args.id!r}: copy it from list_promotions"}
    p = matches[0]
    pages = session.execute(
        select(Document.source_url, Document.title, Document.published_date, Document.full_text)
        .join(PromotionSource, PromotionSource.document_id == Document.id)
        .where(PromotionSource.promotion_id == p.id, Document.is_latest)
        .order_by(Document.source_url)
    ).all()
    row = _row(p, [])
    row.pop("urls", None)
    details = {k: p.raw.get(k) for k in RAW_DETAILS}
    row |= {k: v for k, v in details.items() if v not in (None, [], "")}
    if p.products:
        row["products"] = p.products
    row["pages"] = [
        {
            "url": url,
            "title": title,
            "published": published.isoformat() if published else None,
            "text": text[:PAGE_CHARS] + (" …(cut)" if len(text) > PAGE_CHARS else ""),
        }
        for url, title, published, text in pages
    ]
    return row


def _conditions(f: PromotionFilters) -> list[ColumnElement[bool]]:
    single_day = Promotion.promo_type.in_(SINGLE_DAY_TYPES)
    # The last day a promotion runs: a store event's one day, else end_date (NULL = open-ended).
    last_day = case((single_day, Promotion.start_date), else_=Promotion.end_date)
    conditions = []
    if f.promo_types:
        conditions.append(Promotion.promo_type.in_(f.promo_types))
    if f.active_from:
        conditions.append(or_(last_day.is_(None), last_day >= f.active_from))
    if f.active_to:
        conditions.append(or_(Promotion.start_date.is_(None), Promotion.start_date <= f.active_to))
    if f.start_from:
        conditions.append(Promotion.start_date >= f.start_from)
    if f.start_to:
        conditions.append(Promotion.start_date <= f.start_to)
    if f.store:
        name = f.store.strip().replace("臺", "台")
        conditions.append(
            func.array_to_string(Promotion.stores, "|").icontains(name, autoescape=True)
        )
    if f.city:
        conditions.append(Promotion.cities.contains([f.city]))
    if f.payment_method:
        method = f.payment_method.strip()
        conditions.append(
            func.array_to_string(Promotion.payment_methods, "|").icontains(method, autoescape=True)
        )
    if f.channel:
        conditions.append(Promotion.channels.contains([f.channel]))
    return conditions


def _source_urls(session: Session, ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    urls: dict[uuid.UUID, list[str]] = {}
    rows = session.execute(
        select(PromotionSource.promotion_id, Document.source_url)
        .join(Document, Document.id == PromotionSource.document_id)
        .where(PromotionSource.promotion_id.in_(ids), Document.is_latest)
        .order_by(Document.source_url)
    )
    for promotion_id, url in rows:
        urls.setdefault(promotion_id, []).append(url)
    return urls


def _row(p: Promotion, urls: list[str]) -> dict:
    """One promotion for the model: empty fields left out, to save tokens."""
    row = {
        "id": str(p.id),
        "title": p.title,
        "type": p.promo_type,
        "start": p.start_date.isoformat() if p.start_date else None,
        "end": p.end_date.isoformat() if p.end_date else None,
        "stores": p.stores,
        "cities": p.cities,
        "channels": p.channels,
        "payment_methods": p.payment_methods,
        "min_spend": int(p.min_spend) if p.min_spend is not None else None,
        "summary": p.discount_text,  # store events: date | address | phone
        "urls": urls,
    }
    return {k: v for k, v in row.items() if v not in (None, [], "")}
