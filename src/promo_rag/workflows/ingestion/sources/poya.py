"""Site adapter for www.poya.com.tw: page templates, categories and skip rules.

Everything specific to this website lives here. Adding another source means adding
another adapter that returns the same ParsedPage.

The site has six content templates, all inside `main .contentBox`:
  article      h1.articleTitle + .textEditor (+ .topBox with date and category)
               news, events, policy pages, catalogs
  faq          .faqBox > .faqItem (question in h2.title, answer in .definition)
  store_list   .storeList > .storeItem (one store each)
  two_column   section > .rightBox (member and voucher pages)
  gift         .giftBox .textEditor
"""

import re
from dataclasses import dataclass, field
from datetime import date

from bs4 import BeautifulSoup, Tag

from promo_rag.workflows.ingestion.html_text import (
    block_lines,
    image_urls,
    normalize_line,
    strip_non_content,
)

TITLE_SUFFIX = "-寶雅 POYA"

PROMOTION_CATEGORIES = {"主題活動", "檔期活動", "聯名卡活動", "會員/APP活動"}
CATEGORY_DOC_TYPES = {"門市公告": "store_notice", "得獎公告": "announcement"}
NEWS_PATHS = ("/news-detail/", "/events-detail/")

# Button and link labels that end up inside content blocks.
UI_LINES = {"點擊解鎖更多", "加入社群", "加入數位會員", "#"}

# Pages that are fetched and recorded, but never indexed.
SKIP_URLS = {
    "https://www.poya.com.tw/faq/": "aggregate page: repeats the three FAQ category pages",
}


@dataclass
class ParsedPage:
    title: str
    text: str  # blocks separated by "\n"; FAQ items and store rows by "\n\n"
    template: str
    published_date: date | None = None
    category: str | None = None
    image_urls: list[str] = field(default_factory=list)


def parse_page(html: bytes, url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    title = _title(soup)
    box = soup.select_one("main .contentBox")
    if box is None:
        return ParsedPage(title=title, text="", template="none")

    published, category = _top_box(box)
    images = image_urls(box, url)
    strip_non_content(box)

    if faq := box.select_one(".faqBox"):
        template, text = "faq", _faq_text(faq)
    elif stores := box.select_one(".storeList"):
        template, text = "store_list", _store_list_text(stores)
    elif body := box.select_one(".giftBox .textEditor"):
        template, text = "gift", "\n".join(_content_lines(body))
    elif body := box.select_one(".textEditor"):
        template, text = "article", "\n".join(_drop_repeated_title(_content_lines(body), title))
    elif body := box.select_one(":scope > .rightBox"):
        template, text = "two_column", "\n".join(_content_lines(body))
    else:
        template, text = "fallback", "\n".join(_content_lines(box))

    return ParsedPage(
        title=title,
        text=text,
        template=template,
        published_date=published,
        category=category,
        image_urls=images,
    )


def doc_type_for(category: str | None, url: str) -> str:
    if "/dm-detail/" in url:
        return "catalog"
    if category in PROMOTION_CATEGORIES:
        return "promotion"
    if category in CATEGORY_DOC_TYPES:
        return CATEGORY_DOC_TYPES[category]
    if any(p in url for p in NEWS_PATHS):
        return "other"  # a news category we haven't seen: searchable, but no facts extracted
    return "faq_policy"


def _title(soup: BeautifulSoup) -> str:
    h1 = soup.select_one("main h1.articleTitle")
    if h1 and h1.get_text(strip=True):
        return normalize_line(h1.get_text(" "))
    raw = soup.title.get_text() if soup.title else ""
    return normalize_line(raw.removesuffix(TITLE_SUFFIX).strip())


def _top_box(box: Tag) -> tuple[date | None, str | None]:
    """News/event pages: `.topBox` holds the date (2026.10.07) and the category (# 主題活動)."""
    top = box.select_one(".topBox")
    if top is None:
        return None, None
    published = None
    if (d := top.select_one(".date")) and (m := re.search(r"(\d{4})\.(\d{2})\.(\d{2})", d.text)):
        published = date(int(m[1]), int(m[2]), int(m[3]))
    category = None
    if c := top.select_one(".classTitle"):
        category = normalize_line(c.get_text(" ").replace("#", "")) or None
    return published, category


def _content_lines(el: Tag) -> list[str]:
    return [line for line in block_lines(el) if line not in UI_LINES]


def _drop_repeated_title(lines: list[str], title: str) -> list[str]:
    """Many bodies start by repeating the title; keep it once (in `title`)."""
    while lines and lines[0] == title:
        lines = lines[1:]
    return lines


def _faq_text(faq: Tag) -> str:
    items = []
    for item in faq.select(".faqItem"):
        q_el, a_el = item.select_one("h2.title"), item.select_one(".definition")
        if q_el is None or a_el is None:
            continue
        # "1. 停車資訊 +" → "停車資訊"
        question = re.sub(r"^\d+\.\s*|\s*\+$", "", normalize_line(q_el.get_text(" ")))
        answer = "\n".join(_content_lines(a_el))
        items.append(f"Q: {question}\nA: {answer}")
    return "\n\n".join(items)


def _store_list_text(stores: Tag) -> str:
    """One row per store: "南京三民店 (美妝店) | 南京東路五段118號 | 02-27621211 | 10:00-22:30"."""
    rows = []
    for item in stores.select(".storeItem"):
        name = item.select_one("h3.title")
        kind = item.select_one(".classTitle")
        info = item.select_one(".infoBox")
        if name is None:
            continue
        head = normalize_line(name.get_text(" "))
        if kind is not None:
            head += f" ({normalize_line(kind.get_text(' ').replace('#', ''))})"
        details = _content_lines(info) if info is not None else []
        rows.append(" | ".join([head, *details]))
    return "\n\n".join(rows)
