"""The SQL tools, on the real promotions table: the golden count and list questions."""

import json
from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from promo_rag.agents.tools import TOOLS
from promo_rag.agents.tools.promotions import (
    GetPromotionArgs,
    ListPromotionsArgs,
    PromotionFilters,
    count_promotions,
    get_promotion,
    list_promotions,
)

YEAR = {"start_from": date(2026, 1, 1), "start_to": date(2026, 12, 31)}


def _month(month: int) -> dict:
    end = date(2026, month + 1, 1) if month < 12 else date(2027, 1, 1)
    return {"start_from": date(2026, month, 1), "start_to": date.fromordinal(end.toordinal() - 1)}


@pytest.fixture(scope="module")
def session():
    from promo_rag.db.session import new_session

    s = new_session()
    try:
        count_promotions(s, PromotionFilters())
    except OperationalError:
        pytest.skip("no database")
    yield s
    s.rollback()
    s.close()


def _stores(session, **filters) -> list[str]:
    result = list_promotions(session, ListPromotionsArgs(**filters))
    assert result["total"] == result["shown"]
    return [p["stores"][0] for p in result["promotions"]]


# ---------------------------------------------------------------- golden questions


@pytest.mark.parametrize(
    ("question", "filters", "count"),
    [
        ("count-02", {"promo_types": ["store_opening"], **_month(8)}, 6),
        ("count-03", {"promo_types": ["store_reopening"], **_month(6)}, 5),
        ("count-05", {"promo_types": ["clearance_sale"], **_month(5)}, 4),
        ("count-07", {"promo_types": ["store_opening"], **_month(4)}, 2),
        (
            "count-08",
            {
                "promo_types": ["clearance_sale"],
                "active_from": date(2026, 1, 1),
                "active_to": date(2026, 12, 31),
            },
            16,
        ),
    ],
)
def test_golden_counts(session, question, filters, count):
    assert count_promotions(session, PromotionFilters(**filters))["count"] == count, question


def test_new_stores_in_october(session):
    """list-01"""
    stores = _stores(session, promo_types=["store_opening"], **_month(10))
    assert stores == ["新店中正店", "鳳山和成店", "台中興安店"]  # earliest first


def test_new_stores_in_kaohsiung(session):
    """list-06: 鳳山 and 路竹 are in 高雄市 but not named for it; the address knows."""
    stores = _stores(session, promo_types=["store_opening"], city="高雄市", **YEAR)
    assert set(stores) == {
        "鳳山光遠店",
        "高雄九如店",
        "路竹大社店",
        "高雄茄萣店",
        "鼓山大順店",
        "鳳山和成店",
    }


def test_new_or_reopened_stores_in_taichung(session):
    """list-11"""
    types = ["store_opening", "store_reopening"]
    stores = _stores(session, promo_types=types, city="台中市", **YEAR)
    assert set(stores) == {"大里國光店", "台中東勢店", "台中五權西店", "台中興安店"}


# ---------------------------------------------------------------- filters


def test_a_store_opening_runs_on_its_day_only(session):
    """end_date is NULL for openings: one day, not open-ended."""
    on_the_day = PromotionFilters(store="新店中正店", active_from=date(2026, 10, 17))
    a_week_later = PromotionFilters(store="新店中正店", active_from=date(2026, 10, 24))
    assert count_promotions(session, on_the_day)["count"] == 1
    assert count_promotions(session, a_week_later)["count"] == 0


def test_a_city_also_gets_the_offers_for_all_stores(session):
    """ "现在台北市有什么活动吗" found nothing: offers for all stores have no city."""
    today = {"active_from": date(2026, 10, 12), "active_to": date(2026, 10, 12)}
    running = list_promotions(session, ListPromotionsArgs(limit=50, **today))
    assert running["total"] == running["shown"]
    # Rows leave out empty fields: named stores but no city = can't tell where it applies.
    named_stores_only = [p for p in running["promotions"] if "stores" in p and "cities" not in p]
    taipei = count_promotions(session, PromotionFilters(city="台北市", **today))["count"]
    assert named_stores_only and taipei > 0
    assert taipei == running["total"] - len(named_stores_only)


def test_store_names_match_in_part_and_with_either_tai(session):
    assert _stores(session, store="臺中興安", promo_types=["store_opening"]) == ["台中興安店"]


def test_payment_methods_match_in_part(session):
    result = list_promotions(session, ListPromotionsArgs(payment_method="line pay"))
    assert result["total"] >= 1
    assert all("LINE Pay" in p["payment_methods"] for p in result["promotions"])


def test_the_total_shows_when_the_list_is_cut(session):
    result = list_promotions(session, ListPromotionsArgs(promo_types=["store_opening"], limit=3))
    assert (result["shown"], result["total"] > 3) == (3, True)


def test_rows_leave_out_empty_fields_and_carry_their_pages(session):
    row = list_promotions(session, ListPromotionsArgs(store="新店中正店"))["promotions"][0]
    assert "end" not in row and "payment_methods" not in row
    assert row["urls"] and row["urls"][0].startswith("https://www.poya.com.tw/")


def test_bad_filters_are_rejected_before_any_sql():
    with pytest.raises(ValidationError):
        PromotionFilters(city="Kaohsiung")  # the model must give 高雄市
    with pytest.raises(ValidationError):
        ListPromotionsArgs(limit=500)


# ---------------------------------------------------------------- get_promotion


def _first_id(session, **filters) -> str:
    return list_promotions(session, ListPromotionsArgs(**filters))["promotions"][0]["id"]


def test_get_a_campaign_by_id_prefix_with_its_page(session):
    promotion_id = _first_id(session, payment_method="LINE Pay")
    got = get_promotion(session, GetPromotionArgs(id=promotion_id[:8].upper()))
    assert got["id"] == promotion_id and got["min_spend"] == 650
    assert got["period_text"].startswith("活動期間")  # kept only in raw
    assert "寶雅南港高鐵店" in got["excluded_stores"]
    page = got["pages"][0]
    assert page["url"].endswith("/news-detail/357/") and "LINE POINTS" in page["text"]


def test_get_a_store_notice_shows_its_address_and_phone(session):
    got = get_promotion(session, GetPromotionArgs(id=_first_id(session, store="鼓山大順")))
    assert (got["address"], got["phone"], got["opening_time"]) == (
        "高雄市鼓山區大順一路605-1號",
        "07-5503633",
        "10:00",
    )


def test_an_unknown_id_is_an_error_the_model_can_read(session):
    assert "no promotion" in get_promotion(session, GetPromotionArgs(id="00000000"))["error"]
    with pytest.raises(ValidationError):
        GetPromotionArgs(id="533e")  # too short to be unique


# ---------------------------------------------------------------- LangChain wrappers


def test_tools_are_registered_and_return_json(session):
    assert {"count_promotions", "list_promotions", "get_promotion", "resolve_dates"} <= set(TOOLS)
    out = TOOLS["count_promotions"].invoke(
        {"promo_types": ["store_opening"], "start_from": "2026-08-01", "start_to": "2026-08-31"}
    )
    assert json.loads(out) == {"count": 6, "by_type": {"store_opening": 6}}
