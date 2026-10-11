"""Checks every real POYA page. Runs only where the snapshot exists (it isn't committed).

Set POYA_CORPUS to the snapshot folder, or keep it at ~/Downloads/rag_poya_corpus.
"""

import csv
import os
from datetime import date
from pathlib import Path

import pytest

from promo_rag.workflows.ingestion.actions.parse import MIN_TEXT_CHARS
from promo_rag.workflows.ingestion.sources.poya import parse_page

CORPUS = Path(os.environ.get("POYA_CORPUS", "~/Downloads/rag_poya_corpus")).expanduser()
pytestmark = pytest.mark.skipif(not (CORPUS / "sources.csv").exists(), reason="no local snapshot")

# Pages whose content is only in images, plus the 404. Anything else that parses to
# almost nothing means a template changed or a selector broke.
EXPECTED_EMPTY = {"service", "news-170", "news-338"} | {
    f"dm-{n}" for n in (8, 9, 10, 11, 53, 54, 55, 56)
}


@pytest.fixture(scope="module")
def pages():
    with open(CORPUS / "sources.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["slug"]: parse_page((CORPUS / "raw" / f"{r['slug']}.html").read_bytes(), r["url"])
        for r in rows
    }


def test_no_page_parses_to_almost_nothing(pages):
    empty = {slug for slug, page in pages.items() if len(page.text) < MIN_TEXT_CHARS}
    assert empty == EXPECTED_EMPTY


def test_no_site_chrome_in_any_page(pages):
    for slug, page in pages.items():
        for junk in ("PREV", "NEXT", "Design by iBest", "最新消息 門市查詢"):
            assert junk not in page.text, f"{slug} contains {junk!r}"


def test_campaign_page_keeps_dates_and_rules(pages):
    page = pages["news-361"]
    assert page.published_date == date(2026, 10, 7)
    assert page.category == "主題活動"
    assert "2026/10/07" in page.text and "$5,000" in page.text


@pytest.mark.parametrize("slug", ["faq-6-online", "faq-7-members", "faq-8-stores"])
def test_every_faq_item_becomes_one_question(pages, slug):
    html = (CORPUS / "raw" / f"{slug}.html").read_text(encoding="utf-8")
    items_in_html = html.count('class="faqItem')
    assert items_in_html > 0
    assert pages[slug].text.count("Q: ") == items_in_html
