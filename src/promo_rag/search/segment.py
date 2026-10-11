"""Word splitting for keyword search, shared by ingestion (chunks) and search (questions).

Postgres full-text search splits text at spaces and punctuation, but Chinese has no spaces,
so a whole sentence would become one "word". We split it in Python and store the result
with spaces between words (chunks.search_text).

The page and the question must go through this same function with the same mode, or the
words won't match: jieba splits 回饋金 into 回饋 + 金, so a question kept as 回饋金 misses.

Modes (a measured experiment):
  none     no splitting; Postgres's own rules only
  jieba    jieba's default dictionary plus a POYA word list (poya_words.txt)
  bigram   overlapping 2-character pieces (回饋金 → 回饋 饋金); needs no dictionary
"""

import hashlib
import logging
import re
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

with warnings.catch_warnings():
    # jieba is old code; Python 3.12 warns about its regex strings when it is first compiled.
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba

Segmenter = Literal["none", "jieba", "bigram"]

WORD_LIST = Path(__file__).parent / "poya_words.txt"
CJK_OR_OTHER = re.compile(r"[\u4e00-\u9fff]+|[^\s\u4e00-\u9fff]+")
IS_CJK = re.compile(r"[\u4e00-\u9fff]")
HAS_WORD_CHAR = re.compile(r"\w")
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_numbers(text: str) -> str:
    """5,000 → 5000, so the number stays one word."""
    return THOUSANDS_COMMA.sub("", text)


def segment(text: str, mode: Segmenter = "jieba") -> str:
    text = normalize_numbers(unicodedata.normalize("NFKC", text))
    if mode == "none":
        return text
    tokens = _jieba().lcut(text) if mode == "jieba" else _bigrams(text)
    # Drop punctuation-only pieces; Postgres would drop them anyway.
    return " ".join(t.strip().lower() for t in tokens if HAS_WORD_CHAR.search(t))


def word_list_version() -> str:
    """Changes when poya_words.txt changes, which changes the index fingerprint."""
    return hashlib.sha256(WORD_LIST.read_bytes()).hexdigest()[:12]


@lru_cache
def _jieba() -> jieba.Tokenizer:
    jieba.setLogLevel(logging.WARNING)
    tokenizer = jieba.Tokenizer()
    tokenizer.load_userdict(str(WORD_LIST))
    return tokenizer


def _bigrams(text: str) -> list[str]:
    tokens = []
    for piece in CJK_OR_OTHER.findall(text):
        if IS_CJK.match(piece):
            tokens += (
                [piece] if len(piece) == 1 else [piece[i : i + 2] for i in range(len(piece) - 1)]
            )
        else:
            tokens += re.findall(r"[^\W_]+", piece)  # words and numbers
    return tokens
