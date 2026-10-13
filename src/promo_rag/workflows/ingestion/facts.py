"""Step 3 building blocks: what facts we extract, how we check them, and how they map to rows.

Two schemas, one per doc_type that has facts:
  PromotionFacts     campaign pages; extracted by the LLM
  StoreNoticeFacts   store openings and clearance sales; read by rules (a fixed format),
                     with the LLM as a fallback

The checks run in code after every extraction. Their messages are written for the model,
because on the retry they are sent back to it.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- schemas


class PromotionFacts(BaseModel):
    """Facts from one campaign page. Every value must come from the page."""

    promo_type: Literal[
        "discount", "cashback", "points", "coupon", "gift", "card_offer", "lottery", "other"
    ] = Field(
        description="The main kind of benefit. discount: a lower price at checkout (折, 現折). "
        "cashback: money or a cash voucher given back after buying (回饋金, 現金券). "
        "points: earning or exchanging loyalty points (點數). coupon: a coupon you claim "
        "first and then use (領券, 折價券). gift: a free item with purchase (贈). card_offer: "
        "benefits of one bank or co-branded card. lottery: a prize draw (抽獎). other: none fit."
    )  # fmt: skip
    period_text: str | None = Field(
        description="The campaign period exactly as written on the page, copied character "
        "for character (e.g. '活動期間:2026/10/01-12/31'). It must contain the start and end "
        "dates. null if the page states no period with dates."
    )
    start_date: date | None = Field(
        description="First day of the campaign period. Not a coupon-claim, redemption, "
        "drawing or payback date. null if not stated."
    )
    end_date: date | None = Field(
        description="Last day of the campaign period. null if open-ended or not stated."
    )
    channels: list[Literal["store", "online", "app"]] = Field(
        default_factory=list,
        description="Where the offer applies: store = physical stores (實體門市), online = "
        "the online shop (線上買), app = the POYA app. Empty if not stated.",
    )
    stores: list[str] = Field(
        default_factory=list,
        description="Store names, only if the offer is limited to these stores. Empty means "
        "all stores or not stated. Never list stores where the offer does NOT apply here.",
    )
    excluded_stores: list[str] = Field(
        default_factory=list,
        description="Stores where the offer or its payment method does not apply (不適用).",
    )
    payment_methods: list[str] = Field(
        default_factory=list,
        description="Required payment methods or cards, as named on the page "
        "(e.g. 街口支付, LINE Pay, 國泰世華信用卡). Empty if any payment works.",
    )
    products: list[str] = Field(
        default_factory=list,
        description="Products, categories or brands that qualify, as named on the page.",
    )
    min_spend: int | None = Field(
        description="Minimum spend in one purchase to qualify, in NT$. null if none."
    )
    discount_text: str = Field(
        description="The main benefit in one short sentence, in the page's language and wording."
    )


class StoreNoticeFacts(BaseModel):
    """Facts from one store notice: an opening, a reopening, or a renovation clearance sale."""

    store_name: str = Field(description="The store's name, e.g. 鼓山大順店.")
    notice_type: Literal["store_opening", "store_reopening", "clearance_sale"]
    period_text: str | None = Field(
        description="The opening date or the sale period exactly as written on the page."
    )
    start_date: date | None = Field(description="Opening day, or first day of the sale.")
    end_date: date | None = Field(description="Last day of the sale. null for openings.")
    opening_time: str | None = Field(default=None, description="e.g. 10:00")
    address: str | None = None
    phone: str | None = None


# Store events that happen on one day: start_date is the day, end_date is NULL (not open-ended).
SINGLE_DAY_TYPES = frozenset({"store_opening", "store_reopening"})

# ---------------------------------------------------------------- cities

# Taiwan's 22 cities and counties, written with 台 (the site mixes 台 and 臺).
CITIES = (
    "台北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "苗栗縣", "台中市",
    "彰化縣", "南投縣", "雲林縣", "嘉義市", "嘉義縣", "台南市", "高雄市", "屏東縣",
    "宜蘭縣", "花蓮縣", "台東縣", "澎湖縣", "金門縣", "連江縣",
)  # fmt: skip


def city_from_address(address: str | None) -> str | None:
    """'高雄市鳳山區五甲二路…' → 高雄市. None if it doesn't start with a city or county."""
    head = unicodedata.normalize("NFKC", address or "").strip().replace("臺", "台")[:3]
    return head if head in CITIES else None


# ---------------------------------------------------------------- dates


def resolve_year(month: int, day: int, ref: date) -> date:
    """A date written without a year ('08/22'): the year that puts it closest to `ref`."""
    candidates = []
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:  # Feb 29 in a non-leap year
            pass
    return min(candidates, key=lambda d: abs((d - ref).days))


# ---------------------------------------------------------------- rules for store notices

NOTICE_TAGS = {
    "新店開幕": "store_opening",
    "新櫃開幕": "store_opening",
    "改裝開幕": "store_reopening",
    "改裝出清": "clearance_sale",
}
_DATE = r"(?:\d{4}/)?\d{1,2}/\d{1,2}(?:\(.\))?"
OPENING = re.compile(r"📅\s*(\d{1,2})/(\d{1,2})(?:\(.\))?\s*(\d{1,2}:\d{2})?")
SALE = re.compile(rf"出清(?:日期|期間):\s*(即日起|{_DATE})\s*[~\-]\s*({_DATE})")
PHONE = re.compile(r"📞\s*([\d\-#]+)")
ADDRESS = re.compile(r"📍\s*(.+)")


def parse_store_notice(title: str, text: str, published: date | None) -> StoreNoticeFacts | None:
    """Read a store notice with fixed patterns. None if the page doesn't match them."""
    m = re.match(r"【(.+?)】\s*(.+)", title)
    if not m or m[1] not in NOTICE_TAGS or published is None:
        return None
    notice_type, store_name = NOTICE_TAGS[m[1]], m[2].strip()
    phone = PHONE.search(text)
    address = ADDRESS.search(text)
    common = {
        "store_name": store_name,
        "notice_type": notice_type,
        "phone": phone[1] if phone else None,
        "address": address[1].strip() if address else None,
    }

    if notice_type == "clearance_sale":
        sale = SALE.search(text)
        if not sale:
            return None
        start = published if sale[1] == "即日起" else _parse_date(sale[1], published)
        end = _parse_date(sale[2], start)
        if end < start:
            end = end.replace(year=end.year + 1)
        return StoreNoticeFacts(**common, period_text=sale[0], start_date=start, end_date=end)

    opening = OPENING.search(text)
    if not opening:
        return None
    day = resolve_year(int(opening[1]), int(opening[2]), published)
    return StoreNoticeFacts(
        **common,
        period_text=opening[0],
        start_date=day,
        end_date=None,
        opening_time=opening[3],
    )


def _parse_date(token: str, ref: date) -> date:
    """'2026/05/07(四)' or '06/03(三)' (year from ref)."""
    parts = [int(p) for p in re.findall(r"\d+", token.split("(")[0])]
    if len(parts) == 3:
        return date(*parts)
    return resolve_year(parts[0], parts[1], ref)


# ---------------------------------------------------------------- checks

MAX_DAYS_FROM_PUBLISHED = 400


def check_facts(facts: BaseModel, page_text: str, published: date | None) -> list[str]:
    """Problems with an extraction, as messages the model can act on. Empty list = OK."""
    errors = []
    start, end = getattr(facts, "start_date", None), getattr(facts, "end_date", None)
    period_text = getattr(facts, "period_text", None)

    if (start or end) and not period_text:
        errors.append("start_date/end_date are set but period_text is empty: copy the exact "
                      "text that states the period, or set the dates to null.")  # fmt: skip
    if period_text and not appears_in(period_text, page_text):
        errors.append(f"period_text {period_text!r} does not appear on the page: copy it exactly.")
    elif period_text:
        # The dates must come from the quoted period, not from elsewhere on the page
        # (a coupon deadline or a payback date). "即日起" (from today) means the publish date.
        for name, d in (("start_date", start), ("end_date", end)):
            from_today = name == "start_date" and d == published and "即日起" in period_text
            if d and not from_today and not date_in_text(d, period_text):
                errors.append(f"{name} {d} is not in period_text {period_text!r}: take the "
                              "dates from the campaign period, or set them to null.")  # fmt: skip
    if start and end and start > end:
        errors.append(f"start_date {start} is after end_date {end}.")
    if published:
        for name, d in (("start_date", start), ("end_date", end)):
            if d and abs(d - published) > timedelta(days=MAX_DAYS_FROM_PUBLISHED):
                errors.append(f"{name} {d} is more than a year from the publish date "
                              f"{published}: check the year.")  # fmt: skip
    min_spend = getattr(facts, "min_spend", None)
    if min_spend is not None and not number_in(min_spend, page_text):
        errors.append(f"min_spend {min_spend} does not appear on the page.")
    return errors


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def appears_in(quote: str, page_text: str) -> bool:
    """The quote is on the page, ignoring whitespace and full-width/half-width differences."""
    return _squash(quote) in _squash(page_text)


def date_in_text(d: date, text: str) -> bool:
    """The month and day appear together, as in 2026/10/01, 10/1, 10.01 or 10月1日."""
    pattern = rf"(?<!\d)0?{d.month}\s*[/.\-月]\s*0?{d.day}(?!\d)"
    return re.search(pattern, unicodedata.normalize("NFKC", text)) is not None


def number_in(n: int, page_text: str) -> bool:
    """The number appears on the page, with or without thousands separators (1,000 = 1000)."""
    digits = unicodedata.normalize("NFKC", page_text).replace(",", "")
    return re.search(rf"(?<!\d){n}(?!\d)", digits) is not None


# ---------------------------------------------------------------- facts → promotions row


def dedup_key(title: str, start: date | None, end: date | None) -> str:
    """Same campaign on two pages → same key → one promotions row."""
    norm = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", title).lower())
    return f"{norm}|{start or ''}|{end or ''}"


def promotion_from_campaign(facts: PromotionFacts, title: str) -> dict:
    return {
        "title": title,
        "promo_type": facts.promo_type,
        "start_date": facts.start_date,
        "end_date": facts.end_date,
        "stores": facts.stores,
        "channels": facts.channels,
        "payment_methods": facts.payment_methods,
        "products": facts.products,
        "min_spend": facts.min_spend,
        "discount_text": facts.discount_text,
    }


def promotion_from_store_notice(facts: StoreNoticeFacts, title: str) -> dict:
    details = [facts.period_text, facts.address, facts.phone]
    city = city_from_address(facts.address)
    return {
        "title": title,
        "promo_type": facts.notice_type,
        "start_date": facts.start_date,
        "end_date": facts.end_date,
        "stores": [facts.store_name],
        "cities": [city] if city else [],
        "channels": ["store"],
        "payment_methods": [],
        "products": [],
        "min_spend": None,
        "discount_text": " | ".join(d for d in details if d),
    }


@dataclass(frozen=True)
class Extractor:
    schema: type[BaseModel]
    to_promotion: Callable[[BaseModel, str], dict]
    # Code that reads a fixed format first; None means LLM only.
    rules: Callable[[str, str, date | None], BaseModel | None] | None = None


# doc_type → how to get its facts. Doc types not listed here are search-only.
EXTRACTORS: dict[str, Extractor] = {
    "promotion": Extractor(PromotionFacts, promotion_from_campaign),
    "store_notice": Extractor(StoreNoticeFacts, promotion_from_store_notice, parse_store_notice),
}
# v2: store notices also fill promotions.cities, from the address.
RULES_VERSION = "rules-v2"
