from datetime import date

import pytest

from promo_rag.db.models import Promotion
from promo_rag.workflows.ingestion.facts import (
    EXTRACTORS,
    PromotionFacts,
    StoreNoticeFacts,
    appears_in,
    check_facts,
    date_in_text,
    dedup_key,
    number_in,
    parse_store_notice,
    resolve_year,
)

PUBLISHED = date(2026, 7, 22)


# ---------------------------------------------------------------- the registry


def _sample(schema):
    if schema is PromotionFacts:
        return PromotionFacts(
            promo_type="coupon",
            period_text="2026/10/01-12/31",
            start_date=date(2026, 10, 1),
            end_date=date(2026, 12, 31),
            min_spend=800,
            discount_text="滿$800享88折",
        )
    return StoreNoticeFacts(
        store_name="鼓山大順店",
        notice_type="store_opening",
        period_text="📅08/22(六) 10:00",
        start_date=date(2026, 8, 22),
        end_date=None,
    )


@pytest.mark.parametrize("doc_type", sorted(EXTRACTORS))
def test_every_extractor_fills_real_promotion_columns_including_dates(doc_type):
    """A schema/column name mismatch can leave every date NULL, silently; this catches it."""
    extractor = EXTRACTORS[doc_type]
    row = extractor.to_promotion(_sample(extractor.schema), "title")
    assert set(row) <= set(Promotion.__table__.columns.keys())
    assert row["start_date"] is not None


def test_only_doc_types_with_facts_have_extractors():
    assert set(EXTRACTORS) == {"promotion", "store_notice"}


# ---------------------------------------------------------------- store notice rules


def test_opening_notice_without_a_year():
    text = "📅08/22(六) 10:00\n📞07-5503633\n📍高雄市鼓山區大順一路605-1號\n開幕活動詳情請見店內公告哦!"
    facts = parse_store_notice("【新店開幕】鼓山大順店", text, PUBLISHED)
    assert facts.notice_type == "store_opening"
    assert facts.store_name == "鼓山大順店"
    assert facts.start_date == date(2026, 8, 22)
    assert facts.end_date is None
    assert (facts.opening_time, facts.phone) == ("10:00", "07-5503633")
    assert facts.address == "高雄市鼓山區大順一路605-1號"
    assert check_facts(facts, text, PUBLISHED) == []


@pytest.mark.parametrize(
    ("line", "start", "end"),
    [
        ("出清日期:2026/05/07(四)~06/03(三)", date(2026, 5, 7), date(2026, 6, 3)),
        ("出清期間:2026/06/04(四)-2026/06/24(三)", date(2026, 6, 4), date(2026, 6, 24)),
        ("出清期間:即日起-2026/08/05(三)", PUBLISHED, date(2026, 8, 5)),  # starts on publish day
        ("出清期間:2026/12/20(日)-01/05(二)", date(2026, 12, 20), date(2027, 1, 5)),  # new year
    ],
)
def test_clearance_sale_periods(line, start, end):
    facts = parse_store_notice(
        "【改裝出清】新竹東門店", f"{line}\n封館日期:2026/09/01(二)", PUBLISHED
    )
    assert (facts.notice_type, facts.start_date, facts.end_date) == ("clearance_sale", start, end)


def test_unknown_notice_format_falls_back_to_the_llm():
    assert parse_store_notice("【門市公告】暫停營業", "因颱風暫停營業", PUBLISHED) is None
    assert parse_store_notice("【新店開幕】某店", "開幕日期請見海報", PUBLISHED) is None


def test_resolve_year_picks_the_closest_year():
    assert resolve_year(1, 5, date(2026, 12, 20)) == date(2027, 1, 5)
    assert resolve_year(12, 28, date(2026, 1, 3)) == date(2025, 12, 28)


# ---------------------------------------------------------------- checks

PAGE = "活動期間:2026/10/01-12/31\n單筆消費滿 $1,000 享 88折"


def _facts(**overrides):
    base = {
        "promo_type": "discount",
        "period_text": "活動期間:2026/10/01-12/31",
        "start_date": date(2026, 10, 1),
        "end_date": date(2026, 12, 31),
        "min_spend": 1000,
        "discount_text": "滿$1,000享88折",
    }
    return PromotionFacts(**(base | overrides))


def test_good_facts_pass():
    assert check_facts(_facts(), PAGE, date(2026, 10, 1)) == []


def test_dates_need_evidence_from_the_page():
    errors = check_facts(_facts(period_text=None), PAGE, None)
    assert errors and "period_text is empty" in errors[0]
    errors = check_facts(_facts(period_text="活動期間:2026/10/01-11/30"), PAGE, None)
    assert errors and "does not appear on the page" in errors[0]


def test_impossible_dates_are_caught():
    errors = check_facts(_facts(start_date=date(2027, 1, 1)), PAGE, None)
    assert any("after end_date" in e for e in errors)
    errors = check_facts(_facts(end_date=date(2028, 12, 31)), PAGE, date(2026, 10, 1))
    assert "more than a year" in errors[0]


def test_invented_numbers_are_caught():
    assert "min_spend 800" in check_facts(_facts(min_spend=800), PAGE, None)[0]


def test_matching_ignores_spaces_commas_and_full_width():
    assert appears_in("活動期間 : 2026/10/01-12/31", "活動期間:2026/10/01-12/31")
    assert appears_in("滿＄１,０００", "滿$1,000")
    assert number_in(1000, "滿 $1,000") and not number_in(100, "滿 $1,000")


# ---------------------------------------------------------------- dedup


def test_same_campaign_on_two_pages_gets_one_key():
    start, end = date(2026, 10, 7), date(2026, 11, 3)
    assert dedup_key("十分賺-精選商品滿額贈", start, end) == dedup_key(
        "十分賺 精選商品滿額贈", start, end
    )
    assert dedup_key("十分賺-精選商品滿額贈", start, end) != dedup_key(
        "十分賺-精選商品滿額贈", start, None
    )


def test_dates_must_come_from_the_quoted_period():
    """news-292: the quote was real but dateless, and end_date was the voucher deadline."""
    page = "活動期間每週五~週日,單筆消費滿299元\n(2)使用期限:限於2027/1/31前使用"
    facts = _facts(
        period_text="活動期間每週五~週日",
        start_date=None,
        end_date=date(2027, 1, 31),
        min_spend=299,
    )
    errors = check_facts(facts, page, date(2026, 7, 1))
    assert any("end_date 2027-01-31 is not in period_text" in e for e in errors)


def test_from_today_starts_on_the_publish_date():
    page = "出清期間:即日起-2026/08/31(一)"
    facts = parse_store_notice("【改裝出清】環球青埔店", page, date(2026, 8, 20))
    assert (facts.start_date, facts.end_date) == (date(2026, 8, 20), date(2026, 8, 31))
    assert check_facts(facts, page, date(2026, 8, 20)) == []


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("2026/10/01-12/31", True),
        ("10/1起", True),
        ("10月1日", True),
        ("10.01", True),
        ("110/12", False),
    ],
)
def test_date_in_text(text, ok):
    assert date_in_text(date(2026, 10, 1), text) is ok
