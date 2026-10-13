"""resolve_dates: periods in words → dates, counted from a given today."""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from promo_rag.agents.tools import TOOLS
from promo_rag.agents.tools import dates as dates_module
from promo_rag.agents.tools.dates import DateQuery, resolve_dates

MONDAY = date(2026, 10, 12)


def _span(today=MONDAY, **parts) -> tuple[str, str]:
    r = resolve_dates(DateQuery(**parts), today)
    return r["from"], r["to"]


@pytest.mark.parametrize(
    ("parts", "span"),
    [
        ({"unit": "day"}, ("2026-10-12", "2026-10-12")),  # 今天
        ({"unit": "day", "offset": 1}, ("2026-10-13", "2026-10-13")),  # 明天
        ({"unit": "week"}, ("2026-10-12", "2026-10-18")),  # 這週: Monday to Sunday
        ({"unit": "week", "offset": -1}, ("2026-10-05", "2026-10-11")),  # 上週
        ({"unit": "weekend"}, ("2026-10-17", "2026-10-18")),  # 這個週末
        ({"unit": "month"}, ("2026-10-01", "2026-10-31")),  # 這個月
        ({"unit": "month", "offset": 1}, ("2026-11-01", "2026-11-30")),  # 下個月
        ({"unit": "month", "offset": 3}, ("2027-01-01", "2027-01-31")),  # into next year
        ({"unit": "month", "offset": -10}, ("2025-12-01", "2025-12-31")),  # into last year
        ({"unit": "month", "month": 4}, ("2026-04-01", "2026-04-30")),  # 四月: this year
        ({"unit": "month", "month": 2, "year": 2028}, ("2028-02-01", "2028-02-29")),  # leap year
        ({"unit": "year"}, ("2026-01-01", "2026-12-31")),  # 今年
        ({"unit": "year", "year": 2027}, ("2027-01-01", "2027-12-31")),
    ],
)
def test_periods(parts, span):
    assert _span(**parts) == span


def test_a_week_runs_monday_to_sunday_whatever_day_today_is():
    assert _span(today=date(2026, 10, 18), unit="week") == ("2026-10-12", "2026-10-18")  # Sunday


def test_weekdays_list_each_date_in_the_period():
    """list-05: 這個月禮拜五."""
    r = resolve_dates(DateQuery(unit="month", weekdays=["fri"]), MONDAY)
    assert r["dates"] == ["2026-10-02", "2026-10-09", "2026-10-16", "2026-10-23", "2026-10-30"]
    assert r["today_is"] == "mon"


@pytest.mark.parametrize(
    "parts",
    [
        {"unit": "day", "month": 4},  # a month needs unit month
        {"unit": "week", "year": 2026},
        {"unit": "month", "year": 2026},  # which month?
        {"unit": "month", "month": 13},
        {"unit": "fortnight"},
    ],
)
def test_parts_that_dont_fit_are_rejected(parts):
    with pytest.raises(ValidationError):
        DateQuery(**parts)


def test_the_tool_counts_from_the_pinned_today(monkeypatch):
    monkeypatch.setattr(dates_module, "get_settings", lambda: SimpleNamespace(today=MONDAY))
    out = json.loads(TOOLS["resolve_dates"].invoke({"unit": "month", "offset": 1}))
    assert (out["today"], out["from"], out["to"]) == ("2026-10-12", "2026-11-01", "2026-11-30")
