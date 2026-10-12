"""The golden question set (evals/golden.yaml): questions with the pages that answer them.

Labels are page paths plus a short quote, not chunk ids, so one set works on every corpus
version: a chunk is relevant if it comes from a labeled page and contains that page's quote.
"""

import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

GOLDEN_PATH = Path(__file__).resolve().parents[3] / "evals" / "golden.yaml"
SITE = "https://www.poya.com.tw"

Category = Literal["rule", "date", "list", "count", "unanswerable", "multi_turn"]
ID_PREFIX = {
    "rule": "rule-",
    "date": "date-",
    "list": "list-",
    "count": "count-",
    "unanswerable": "none-",
    "multi_turn": "multi-",
}
# A long quote is more likely to be cut in two by a chunk boundary.
MAX_QUOTE_CHARS = 40


def normalize(text: str) -> str:
    """NFKC (as parse does) and no whitespace, so quotes match however a chunk was packed."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


class Evidence(BaseModel):
    grade: Literal[1, 2]  # 2 = answers the question, 1 = helps
    quote: list[str] = Field(min_length=1)  # a chunk must contain one of these

    @field_validator("quote", mode="before")
    @classmethod
    def _one_or_many(cls, value):
        return [value] if isinstance(value, str) else value

    @field_validator("quote")
    @classmethod
    def _short(cls, quotes: list[str]) -> list[str]:
        for q in quotes:
            if not 0 < len(normalize(q)) <= MAX_QUOTE_CHARS:
                raise ValueError(f"quote must be 1-{MAX_QUOTE_CHARS} characters: {q!r}")
        return quotes


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Question(BaseModel):
    id: str
    category: Category
    lang: Literal["zh", "en"]
    question: str
    history: list[Turn] = []
    relevant: dict[str, Evidence]  # page path → grade and quote
    answer: str | None
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if not self.id.startswith(ID_PREFIX[self.category]):
            raise ValueError(f"{self.id}: id should start with {ID_PREFIX[self.category]!r}")
        if self.category == "unanswerable":
            if self.relevant or self.answer is not None:
                raise ValueError(f"{self.id}: unanswerable needs empty relevant and answer: null")
        else:
            if all(e.grade != 2 for e in self.relevant.values()):
                raise ValueError(f"{self.id}: needs at least one page graded 2")
            if not self.answer:
                raise ValueError(f"{self.id}: needs an answer")
        if (self.category == "multi_turn") != bool(self.history):
            raise ValueError(f"{self.id}: history is for multi_turn questions only, and required")
        roles = [t.role for t in self.history]
        if roles != ["user", "assistant"] * (len(roles) // 2) or len(roles) % 2:
            raise ValueError(f"{self.id}: history must alternate user, assistant")
        for path in self.relevant:
            if not path.startswith("/"):
                raise ValueError(f"{self.id}: page path must start with '/': {path}")
        return self

    def relevant_urls(self) -> dict[str, Evidence]:
        return {SITE + path: evidence for path, evidence in self.relevant.items()}

    def chunk_grade(self, source_url: str, chunk_text: str) -> int:
        """The grade of one search result: 0 unless its page is labeled and it holds a quote."""
        evidence = self.relevant_urls().get(source_url)
        if evidence is None:
            return 0
        text = normalize(chunk_text)
        return evidence.grade if any(normalize(q) in text for q in evidence.quote) else 0


class GoldenSet(BaseModel):
    as_of: date
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "GoldenSet":
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self


def load_golden(path: Path = GOLDEN_PATH) -> GoldenSet:
    return GoldenSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
