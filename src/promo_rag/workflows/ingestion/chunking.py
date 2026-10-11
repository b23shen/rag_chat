"""Cutting a document into chunks. Pure functions, no database.

  articles      split into sentences (。！？；!?; and line ends), then pack whole sentences
                up to the token budget; the next chunk repeats the last ~100 tokens
  FAQ pages     one chunk per question (each "Q: …" / "A: …" block); the question goes in the header
  store list    pack whole rows

Every chunk starts with a header line, e.g. "十分賺-… | 2026-10-07 ~ 2026-11-03", so a chunk
cut from the middle of a page still says which promotion it belongs to.
"""

import math
import re
from dataclasses import dataclass
from datetime import date

from promo_rag.workflows.ingestion.index_config import IndexConfig

SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")


@dataclass(frozen=True)
class Unit:
    """A sentence or a block, and what separated it from the next one."""

    text: str
    sep: str = "\n"


@dataclass(frozen=True)
class HeaderFacts:
    """What the header line says about a document (from its promotion, if it has one)."""

    title: str
    start: date | None = None
    end: date | None = None
    stores: tuple[str, ...] = ()
    single_day: bool = False  # store openings: one date, not an open-ended period


@dataclass(frozen=True)
class ChunkDraft:
    index: int
    text: str  # header line + body: what the LLM reads and what is embedded
    token_count: int


def estimate_tokens(text: str, cfg: IndexConfig) -> int:
    return math.ceil(len(text) / cfg.chars_per_token)


def header_line(facts: HeaderFacts, extra: str | None = None) -> str:
    parts = [facts.title]
    if facts.start and (facts.single_day or facts.end == facts.start):
        parts.append(f"{facts.start}")
    elif facts.start or facts.end:
        parts.append(f"{facts.start or ''} ~ {facts.end or ''}".strip())
    stores = [s for s in facts.stores if s not in facts.title]  # don't repeat the title
    if stores:
        parts.append(", ".join(stores))
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def split_sentences(text: str) -> list[Unit]:
    units = []
    for line in text.split("\n"):
        pieces = [p.strip() for p in SENTENCE_END.split(line.strip()) if p.strip()]
        for i, piece in enumerate(pieces):
            units.append(Unit(piece, "\n" if i == len(pieces) - 1 else ""))
    return units


def pack(units: list[Unit], budget: int, overlap: int, cfg: IndexConfig) -> list[str]:
    """Group units into bodies of at most `budget` tokens; each new body repeats up to
    `overlap` tokens from the end of the previous one, so a fact on a border is whole in one."""
    # Count the separator too: the newlines between sentences end up in the chunk text.
    sized = [(u, estimate_tokens(u.text + u.sep, cfg)) for u in _split_long(units, budget, cfg)]
    bodies: list[list[Unit]] = []
    current: list[tuple[Unit, int]] = []
    for unit, tokens in sized:
        if current and sum(t for _, t in current) + tokens > budget:
            bodies.append([u for u, _ in current])
            tail: list[tuple[Unit, int]] = []
            for u, t in reversed(current):
                if sum(x for _, x in tail) + t > overlap:
                    break
                tail.insert(0, (u, t))
            while tail and sum(t for _, t in tail) + tokens > budget:
                tail.pop(0)
            current = tail
        current.append((unit, tokens))
    if current:
        bodies.append([u for u, _ in current])
    return ["".join(u.text + u.sep for u in body).strip() for body in bodies]


def chunk_document(full_text: str, facts: HeaderFacts, cfg: IndexConfig) -> list[ChunkDraft]:
    blocks = [b.strip() for b in full_text.split("\n\n") if b.strip()]
    pieces: list[tuple[str, str]] = []  # (header, body)

    if blocks and all(b.startswith("Q: ") for b in blocks):
        # FAQ: one chunk per question; a long answer is packed into several, each with the question.
        for block in blocks:
            question = block.split("\n", 1)[0].removeprefix("Q: ").strip()
            header = header_line(facts, question) if cfg.header else ""
            for body in pack(split_sentences(block), _budget(header, cfg), cfg.overlap_tokens, cfg):
                pieces.append((header, body))
    else:
        header = header_line(facts) if cfg.header else ""
        units = (
            [Unit(b, "\n\n") for b in blocks]  # store rows: keep each row whole
            if len(blocks) > 1
            else split_sentences(full_text)
        )
        for body in pack(units, _budget(header, cfg), cfg.overlap_tokens, cfg):
            pieces.append((header, body))

    drafts = []
    for i, (header, body) in enumerate(pieces):
        text = f"{header}\n{body}" if header else body
        drafts.append(ChunkDraft(index=i, text=text, token_count=estimate_tokens(text, cfg)))
    return drafts


def _budget(header: str, cfg: IndexConfig) -> int:
    """Tokens left for the body once the header line is in."""
    return max(cfg.chunk_tokens - estimate_tokens(header, cfg) - 1, cfg.chunk_tokens // 2)


def _split_long(units: list[Unit], budget: int, cfg: IndexConfig) -> list[Unit]:
    """A unit longer than the budget (a run-on paragraph) is cut into budget-sized pieces."""
    max_chars = int(budget * cfg.chars_per_token)
    out = []
    for u in units:
        if len(u.text) <= max_chars:
            out.append(u)
            continue
        pieces = [u.text[i : i + max_chars] for i in range(0, len(u.text), max_chars)]
        out += [Unit(p, "") for p in pieces[:-1]] + [Unit(pieces[-1], u.sep)]
    return out
