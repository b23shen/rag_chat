"""resolve_dates: turn "this month", "next week" or "Fridays in October" into real dates.

The model reads the words and fills the parts (unit, offset, month, year, weekdays); this
code does the date arithmetic, which models often get wrong (the last day of a month, the
date of "this Friday"). The result feeds the date filters of count_promotions and
list_promotions.

"Today" is the store's date, in Taiwan. Set TODAY in .env to pin it, or wrap a run in
pinned_today(d) (the answer eval pins the golden set's as_of date), so "now" and "this month"
mean the same days on every run.
"""

import calendar
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from promo_rag.config import get_settings

STORE_TZ = ZoneInfo("Asia/Taipei")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # date.weekday() order
Weekday = Literal[WEEKDAYS]


_pinned: date | None = None


def today() -> date:
    return _pinned or get_settings().today or datetime.now(STORE_TZ).date()


@contextmanager
def pinned_today(day: date) -> Iterator[None]:
    """Inside the block, today() is `day`, for the tools and the chat prompt alike."""
    global _pinned
    before, _pinned = _pinned, day
    try:
        yield
    finally:
        _pinned = before


class DateQuery(BaseModel):
    """A period described in words, split into parts."""

    unit: Literal["day", "week", "weekend", "month", "year"] = Field(
        description="The kind of period. week: Monday to Sunday. weekend: Saturday and Sunday."
    )
    offset: int = Field(
        default=0,
        ge=-60,
        le=60,
        description="Counted from today: 0 = this (today, this week, this month, 現在, 這個月), "
        "1 = next (明天, 下週, 下個月), -1 = last (昨天, 上週, 上個月). Not used with month or year.",
    )
    month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="A named month (October, 十月 = 10). Only with unit month.",
    )
    year: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="A named year (2026). With unit year, or with month (default: this year).",
    )
    weekdays: list[Weekday] | None = Field(
        default=None,
        description="Only these days of the week in the period, e.g. Fridays this month: "
        "unit month, weekdays [fri]. The result then lists each date.",
    )

    @model_validator(mode="after")
    def _parts_fit(self):
        if self.month is not None and self.unit != "month":
            raise ValueError("month is only for unit month")
        if self.year is not None and self.unit not in ("month", "year"):
            raise ValueError("year is only for unit month or year")
        if self.year is not None and self.unit == "month" and self.month is None:
            raise ValueError("with unit month, a year needs a month too")
        return self


def resolve_dates(query: DateQuery, today: date) -> dict:
    """{"today", "today_is", "from", "to"}, plus "dates" when weekdays are given. Dates are
    ISO strings, ready for the date filters of the promotion tools."""
    start, end = _period(query, today)
    result = {
        "today": today.isoformat(),
        "today_is": WEEKDAYS[today.weekday()],
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    if query.weekdays:
        wanted = {WEEKDAYS.index(d) for d in query.weekdays}
        days = (start + timedelta(n) for n in range((end - start).days + 1))
        result["dates"] = [d.isoformat() for d in days if d.weekday() in wanted]
    return result


def _period(q: DateQuery, today: date) -> tuple[date, date]:
    if q.unit == "day":
        day = today + timedelta(days=q.offset)
        return day, day
    if q.unit in ("week", "weekend"):
        monday = today - timedelta(days=today.weekday()) + timedelta(weeks=q.offset)
        if q.unit == "weekend":
            return monday + timedelta(days=5), monday + timedelta(days=6)
        return monday, monday + timedelta(days=6)
    if q.unit == "month":
        if q.month is not None:
            year, month = q.year or today.year, q.month
        else:
            year, month = divmod(today.year * 12 + today.month - 1 + q.offset, 12)
            month += 1
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    year = q.year or today.year + q.offset
    return date(year, 1, 1), date(year, 12, 31)
